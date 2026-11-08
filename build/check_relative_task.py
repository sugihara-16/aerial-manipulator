"""Evaluate object-relative control on a task-generated structure.

Saved plans are development replay inputs, never an online structure catalogue.
Physics includes every authored robot collision geometry. Measured mode applies
the old command during computation; integration cost is separate from control.
"""
from amsrr_det.collision_policy import configure as configure_collision
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from build.load_saved_plan import load_saved_plan
from build.mit_drive import dock_mit_model
from build.generated_grasp_stage import align_reference_to_object
from build.relative_task_control import ObjectRelativeControl
from build.control_latency import ControlCycle
from build.async_allocation import AsyncAllocation
from amsrr_det.task_audit import task_checks
from amsrr_det.fast_task_audit import FastTaskAudit
from amsrr_det.task_geometry import prepare_opening, transport_height


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))


class TaskReferences:
    """Task-specific contact roles; the underlying control law never switches."""
    phases = ['APPROACH', 'LOAD', 'LIFT', 'CARRY', 'DESCEND', 'UNLOAD', 'OPEN', 'RETREAT']
    durations = [5., 2., 5., 8., 6., 2., 4., 3.]

    def __init__(self, plan, plant, closed):
        self.plan, self.m = plan, plant.mj
        self.open = plan.compact_open.copy()
        self.closed = closed.copy()
        self.lift_height = transport_height(plan.shape)
        self.oi = int(self.m.jnt_qposadr[self.m.joint('object').id])
        self.source = self.closed[self.oi:self.oi+7].copy()
        self.destination = np.r_[plan.destination[:3], plan.destination[[6, 3, 4, 5]]]
        self.index, self.progress, self.rate = 0, 0., 1.
        self.events = [dict(time_s=0., phase='APPROACH')]
        self.last_time = 0.
        self.complete = False
        self.delta = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, self.delta, 1., self.open, self.closed)
        self.retreat = np.zeros(self.m.nv)
        kd=mujoco.MjData(self.m);kd.qpos[:]=self.open;mujoco.mj_kinematics(self.m,kd)
        links=[f'm{n}/'+('yaw_dock_mech' if a.startswith('yaw') else 'pitch_dock_mech')+a[-1] for n,a in plan.surfaces]
        ids=[i for i in range(self.m.ngeom) if any((self.m.geom(i).name or '').startswith(link+'/') for link in links)]
        rotations=kd.geom_xmat[ids].reshape(-1,3,3)
        bounds=self.m.geom_aabb[ids]
        centers=kd.geom_xpos[ids]+np.einsum('nij,nj->ni',rotations,bounds[:,:3])
        pad_bottom=np.min(centers[:,2]-np.einsum('ni,ni->n',abs(rotations[:,2]),bounds[:,3:]))
        oi=self.m.geom('object').id;Ro=kd.geom_xmat[oi].reshape(3,3);bo=self.m.geom_aabb[oi]
        object_top=(kd.geom_xpos[oi]+Ro@bo[:3])[2]+abs(Ro[2])@bo[3:]
        self.retreat_height=max(.15,float(object_top+.04-pad_bottom))
        self.durations=list(self.durations);self.durations[-1]=max(3.,1.875*self.retreat_height/.25)
        self.durations[2]*=np.sqrt(self.lift_height/.18)
        self.durations[4]*=np.sqrt((self.lift_height+.001)/.181)
        for j in np.flatnonzero(self.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            if self.m.joint(int(j)).name != 'object':
                self.retreat[self.m.jnt_dofadr[j]+2] = self.retreat_height

    @property
    def phase(self):
        return self.phases[self.index]

    def update(self, now, q, v, status):
        dt = max(0., now-self.last_time)
        self.last_time = now
        gamma = 1.
        if status and self.phase in ('LIFT', 'CARRY', 'DESCEND'):
            gap = max(0., max(status['pad_gaps']))
            error = status['relative_position_error_m']/.025
            gamma = max(.1, 1/(1+(status['object_position_error_m']/.04)**2))
        self.rate += min(1., dt/.25)*(gamma-self.rate)
        self.progress = min(1., self.progress + dt*self.rate/self.durations[self.index])
        if self.progress >= 1.:
            ready = True
            if ready:
                if self.index == len(self.phases)-1:
                    self.complete = now-self.events[-1]['time_s'] >= self.durations[-1]+2.
                else:
                    self.index += 1
                    self.progress = 0.
                    self.events.append(dict(time_s=float(now), phase=self.phase))
        s = smooth(self.progress)
        ds = 30*self.progress**2*(1-self.progress)**2*self.rate/self.durations[self.index]
        shape = self.closed.copy()
        shape_velocity = np.zeros(self.m.nv)
        target = self.source.copy()
        velocity = np.zeros(6)
        fraction, squeeze = 1., 1.
        if self.phase == 'APPROACH':
            shape = self.open.copy()
            mujoco.mj_integratePos(self.m, shape, self.delta, float(s))
            shape_velocity = self.delta*ds
            fraction, squeeze = 0., s
        elif self.phase == 'LOAD':
            fraction = s
        elif self.phase == 'LIFT':
            target[2] += self.lift_height*s
            velocity[2] = self.lift_height*ds
            fraction = 1.
        elif self.phase == 'CARRY':
            target[:3] = self.source[:3] + s*(self.destination[:3]-self.source[:3]) + [0., 0., self.lift_height]
            target[3:] = self.destination[3:]
            velocity[:3] = ds*(self.destination[:3]-self.source[:3])
        elif self.phase == 'DESCEND':
            target = self.destination.copy()
            target[2] += self.lift_height*(1-s)-.001*s
            velocity[2] = -(self.lift_height+.001)*ds
        else:
            target = self.destination.copy()
            target[2] -= .001
            if self.phase == 'UNLOAD':
                fraction, squeeze = 1-s, 1-s
            else:
                fraction, squeeze = 0., 0.
                opening = s if self.phase == 'OPEN' else 1.
                mujoco.mj_integratePos(self.m, shape, self.delta, -float(opening))
                shape_velocity = -self.delta*ds if self.phase == 'OPEN' else shape_velocity
                if self.phase == 'RETREAT':
                    mujoco.mj_integratePos(self.m, shape, self.retreat, float(s))
                    shape_velocity = self.retreat*ds
        return shape, target, velocity, shape_velocity, fraction, squeeze


class References(TaskReferences):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.release_anchor=None
 def update(self,now,q,v,status):
  shape,target,vel,svel,fraction,squeeze=super().update(now,q,v,status)
  if self.phase not in ('OPEN','RETREAT'):return shape,target,vel,svel,fraction,squeeze
  if self.release_anchor is None:
   closed=align_reference_to_object(self.m,self.closed,q);opened=align_reference_to_object(self.m,self.open,q)
   self.release_delta=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,self.release_delta,1.,closed,opened)
   Rclosed=np.empty((3,3));Ractual=np.empty((3,3));mujoco.mju_quat2Mat(Rclosed.ravel(),closed[3:7]);mujoco.mju_quat2Mat(Ractual.ravel(),q[3:7]);self.release_delta[3:6]=Ractual.T@Rclosed@self.release_delta[3:6]
   self.release_anchor=q.copy();self.release_anchor[self.oi:self.oi+7]=target
  s=smooth(self.progress);ds=30*self.progress**2*(1-self.progress)**2*self.rate/self.durations[self.index]
  shape=self.release_anchor.copy();opening=s if self.phase=='OPEN'else 1.
  mujoco.mj_integratePos(self.m,shape,self.release_delta,float(opening))
  svel=self.release_delta*ds if self.phase=='OPEN'else np.zeros(self.m.nv)
  if self.phase=='RETREAT':
   mujoco.mj_integratePos(self.m,shape,self.retreat,float(s));svel=self.retreat*ds
  return shape,target,vel,svel,fraction,squeeze


def main(argv=None,*,observer=None,plan_only=False,integration=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='task JSON, or a saved generated-plan directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('--structure-workers', type=int, default=32)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--worker-cpu', type=int, help='optional CPU for allocation worker')
    parser.add_argument('--timing', choices=['measured', 'ideal'], default='measured')
    parser.add_argument('--duration', type=float, default=70.)
    parser.add_argument('--no-relative-feedback', action='store_true')
    parser.add_argument('--collision', choices=['shape-pairs'], default='shape-pairs')
    parser.add_argument('--timestep', type=float, default=.000125)
    args = parser.parse_args(argv)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    fresh = args.source.is_file()
    if fresh:
        from build.select_structure import select_structure
        from amsrr_det.evidence import save_plan
        request = json.loads(args.source.read_text())
        try:
            plan, prepared, selection = select_structure(
                request, workers=args.structure_workers, seed=args.seed,
                task_wrench=True, generator='task')
        except ValueError as error:
            (out/'selection.json').write_text(json.dumps(getattr(error, 'evidence', {})))
            (out/'result.json').write_text(json.dumps(dict(task_success=False,
                reason='planning_failed:'+str(error), initial_planning_s=time.perf_counter()-started)))
            return
        (out/'selection.json').write_text(json.dumps(selection))
        save_plan(plan, out/'generated_plan', 'fresh_task_generation')
        (out/'generated_plan/request.json').write_text(json.dumps(request))
        closed = prepared['qair'].copy()
        for joint in np.flatnonzero(plan.plant.mj.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            closed[plan.plant.mj.jnt_qposadr[joint]+2] -= .15
        np.savez(out/'endpoint.npz', q=closed)
        planning_s = time.perf_counter()-started
        if planning_s > 15.:
            (out/'result.json').write_text(json.dumps(dict(task_success=False,
                reason='initial_planning_deadline', initial_planning_s=planning_s)))
            return
    else:
        plan, request = load_saved_plan(args.source/'generated_plan')
        closed = np.load(args.source/'endpoint.npz')['q'].copy()
        planning_s = None
    if plan_only:return dict(task_success=None,reason="structure_generated",planning_s=planning_s)
    plant = dock_mit_model(plan.plant, kp=200.)
    m = plant.mj
    m.opt.timestep = args.timestep
    configure_collision(m)
    m.opt.impratio = 10000.
    m.opt.ccd_tolerance = 1e-12
    m.opt.ccd_iterations = 1000
    links = [f'm{n}/'+('yaw_dock_mech' if a.startswith('yaw') else 'pitch_dock_mech')+a[-1]
             for n, a in plan.surfaces]
    controller = ObjectRelativeControl(plant, closed, links, object_shape=plan.shape)
    controller.relative_feedback = not args.no_relative_feedback
    plan.compact_open,initial_command,opening_info=prepare_opening(controller,closed,plant.configuration(plan.open_q))
    (out/'opening_preparation.json').write_text(json.dumps(opening_info,indent=2))
    references = References(plan, plant, closed)
    (out/'clearance_reference.json').write_text(json.dumps(dict(lift_height_m=references.lift_height,position_allowance_m=.03,rotation_allowance_rad=.2,required_clearance_m=.10)))
    drive = controller.drive
    d = mujoco.MjData(m)
    d.qpos[:] = references.open
    d.ctrl[:drive.nj] = (drive.kp*d.qpos[drive.qi]+np.array([j['effort_nm'] for j in plan.trims[0]['joints']]))/drive.gain[:drive.nj]
    d.ctrl[drive.nj:] = plan.trims[0]['thrust']
    if initial_command is not None:d.ctrl[:]=initial_command
    mujoco.mj_forward(m, d)
    controller.last_command=d.ctrl.copy()
    task_clock=0.;physical=plant
    if integration is not None:
        physical,d=integration.start(plant,d);m=physical.mj;task_clock=float(d.time)
        from amsrr_det.task_sequence import BlendedControl
        controller=BlendedControl(controller,d.ctrl,integration.map)
    allocation = AsyncAllocation(controller, args.worker_cpu)
    (out/'physical.xml').write_text(physical.xml)
    audit = FastTaskAudit(physical, links, plan.shape.size, plan.support_top, shape=plan.shape, clearance_period=.1)
    (out/'request.json').write_text(json.dumps(request, indent=2))
    (out/'config.json').write_text(json.dumps(dict(timing=args.timing, period_s=.01, timestep_s=m.opt.timestep,
        all_robot_collisions=True, collision=args.collision, auxiliary_stops=False, duration_s=args.duration, source=str(args.source.resolve()),
        relative_feedback=controller.relative_feedback, asynchronous=True, intertick_publication=True, worker_cpu=args.worker_cpu), indent=2))
    np.savez(out/'initial.npz', q=d.qpos, v=d.qvel, ctrl=d.ctrl)
    if observer is not None and integration is None:observer.begin(m,plant.xml)
    calls, trace = [], []
    failure = None
    reason = 'task_timeout'

    publish_events=[]
    poll_cpu=0.
    def plain_advance(steps):
        nonlocal failure
        failure=(integration.advance_audited(audit,steps,carried=references.phase=='CARRY') if integration is not None else audit.advance(d,steps,carried=references.phase=='CARRY'))
        return failure
    def advance(steps):
        nonlocal failure,poll_cpu
        end=float(d.time)+steps*m.opt.timestep
        while d.time < end-1e-10:
            observed=float(d.time);start=time.perf_counter()
            applied=allocation.poll(observed)
            command=allocation.command.copy() if applied else None
            elapsed=time.perf_counter()-start;poll_cpu+=elapsed
            if applied:
                previous=d.ctrl.copy()
                delay_steps=max(1,int(np.ceil(elapsed/m.opt.timestep-1e-10)))+1
                if plain_advance(delay_steps):return failure
                apply_start=time.perf_counter()
                d.ctrl[:]=command
                elapsed+=time.perf_counter()-apply_start
                extra=max(0,int(np.ceil(elapsed/m.opt.timestep-1e-10))-delay_steps)
                if extra:
                    d.ctrl[:]=previous
                    if plain_advance(extra):return failure
                    d.ctrl[:]=command
                allocation.records[-1]['application_s']=float(d.time)
                publish_events.append(dict(observation_s=observed,calculation_s=elapsed,application_s=float(d.time),source_observation_s=allocation.records[-1]['observation_s']))
            left=max(0,round((end-d.time)/m.opt.timestep))
            count=min(left,max(1,round(.001/m.opt.timestep)))
            if count and plain_advance(count):return failure
        return failure

    # Refcounted temporary arrays are released immediately. Defer cyclic
    # garbage collection, whose unbounded graph walk is not a control job.
    collect_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    if observer is not None:observer.ready()
    if integration is not None:
        integration.resume(plain_advance);task_clock=float(d.time)
    try:
        while d.time-task_clock < args.duration and not references.complete:
            poll_cpu=0.
            began = time.perf_counter()
            cycle = ControlCycle(m, d, .01, measured=args.timing == 'measured')
            observed_q,observed_v=(integration.map.observe(cycle.q,cycle.v) if integration is not None else (cycle.q,cycle.v))
            shape, target, velocity, shape_velocity, fraction, squeeze = references.update(d.time-task_clock, observed_q, observed_v, controller.last)
            command = allocation.update(float(d.time), references.phase,
                (observed_q, observed_v, shape, target),
                dict(observation_time=float(d.time),object_velocity=velocity, shape_velocity=shape_velocity,
                     payload_fraction=fraction, squeeze=squeeze, damping_scale=4. if references.phase in ('UNLOAD', 'OPEN') else 1., tracking_fraction=1. if references.phase in ('UNLOAD', 'OPEN') else fraction,
                     track_object=references.phase not in ('OPEN', 'RETREAT')), observed_wall=began)
            if observer is not None:observer.update(d,references.phase)
            elapsed = time.perf_counter()-began
            cycle.wait_for_result(elapsed, plain_advance)
            if not failure:
                cycle.apply(command)
                if allocation.records and allocation.records[-1].get('application_tick_s')==cycle.observed_at:
                    allocation.records[-1]['application_s']=float(d.time)
                advance(cycle.remaining_steps())
            record=cycle.record(references.phase)
            record['heartbeat_calculation_s']=record['calculation_s']
            record['intra_period_publication_s']=poll_cpu
            record['publication_cpu_s']=record['calculation_s']+poll_cpu
            record['skipped_ticks']=max(record['skipped_ticks'],int(np.ceil((d.time-cycle.observed_at)/.01-1e-8))-1)
            record['next_observation_s']=float(d.time)
            calls.append(record)
            if len(calls) % 10 == 0:
                trace.append(dict(time_s=float(d.time), phase=references.phase, progress=float(references.progress),
                                  object_pose=d.qpos[int(m.jnt_qposadr[m.joint('object').id]):int(m.jnt_qposadr[m.joint('object').id])+7].tolist(), **controller.last))
            if len(calls) % 500 == 0:
                (out/'progress.json').write_text(json.dumps(dict(time_s=d.time, phase=references.phase, progress=references.progress, **controller.last)))
            if time.perf_counter()-started > 600.:
                failure = 'wall_time_limit'
            if failure:
                reason = failure
                break
    finally:
        allocation.close()
        if collect_enabled:
            gc.enable()
    checks = task_checks(physical, plan, d, audit)
    checks['transport_clearance'] = bool(np.isfinite(audit.min_clearance_carried) and audit.min_clearance_carried >= .10)
    success = bool(not failure and all(checks.values()))
    if references.complete:
        reason = 'task_completed' if success else 'task_goal_not_reached'
    times = np.array([r['publication_cpu_s'] for r in calls])
    result = dict(task_success=success, reason=reason, modules=plan.plant.meta['modules'],
                  task_checks=checks, audit=audit.result(), events=references.events,
                  phase=references.phase, simulation_s=float(d.time), wall_s=time.perf_counter()-started,
                  scope=('fresh_generation_assembly_task' if integration is not None else 'fresh_generation_spawn_task') if fresh else 'saved_generated_structure_control_replay',
                  task_start_s=task_clock,
                  initial_planning_s=planning_s,
                  control=dict(calls=len(calls), mean_ms=float(times.mean()*1000), p99_ms=float(np.quantile(times, .99)*1000),
                               maximum_ms=float(times.max()*1000), misses=int(np.count_nonzero(times >= .01))))
    result['allocation'] = dict(calls=len(allocation.records), held_ticks=allocation.held_ticks, maximum_ms=max((r['calculation_s'] for r in allocation.records),default=0.)*1000)
    (out/'publication_events.json').write_text(json.dumps(publish_events,indent=2))
    (out/'allocation_calls.json').write_text(json.dumps(allocation.records,indent=2))
    np.savez(out/'final.npz', q=d.qpos, v=d.qvel, ctrl=d.ctrl)
    for name, value in [('result', result), ('trace', trace), ('control_calls', calls)]:
        (out/(name+'.json')).write_text(json.dumps(value, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
