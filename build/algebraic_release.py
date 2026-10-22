"""Supported unloading plus geometry-derived empty flight.

Generation uses equilibrium and geometry, not trial task rollouts. Inputs are
observed poses/velocities and previous commands. Contact forces are audit-only.
"""
import time
import numpy as np
import mujoco
from build.unloaded_flight import UnloadedFlight
from build.unloaded_hover import command as gravity_command
from amsrr_det.grasp_control import GraspReferences
from amsrr_det.retreat import FixedRetreat,task_clearance_gap,rest_to_rest_durations
from amsrr_det.release import check_opening_clearance
from amsrr_det.numerics import quintic_reference
import amsrr_det.coupled_flight as coupled
from build.native_drive_interval_avx_fixed import NativeDriveInterval
from build.native_bvls.solver import solve

coupled.DrivePrediction=NativeDriveInterval
coupled.lsq_linear=solve


class SupportedWithdrawal:
    def __init__(self,plan,plant,q):
        self.plan,self.plant=plan,plant;self.m=plant.mj
        self.local=UnloadedFlight(plant,q);self.query=mujoco.MjData(self.m)
        self.elapsed=0.;self.ready=False;self.complete=False;self.planning_s=0.;self.info={}

    def prepare(self,q,v,previous):
        began=time.perf_counter();plan=self.plan;p=self.plant;m=self.m;local=self.local
        self.query.qpos[:]=q;self.query.qvel[:]=v;qpin,_=p.observation(self.query)
        grasp=GraspReferences(p.pin,p.meta,qpin,qpin,plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape)
        oi=p.meta['object']['q'];gap=task_clearance_gap(plan.shape,qpin[oi:oi+7],plan.destination)
        obstacles=[(m.geom(name).pos.copy(),m.geom(name).size.copy())for name in ['support','destination_support']]
        self.path=FixedRetreat(grasp,qpin,opening_gap_m=.01,final_gap_m=gap,obstacles=obstacles,maximum_material_step=.01)
        clearance=check_opening_clearance(p.meta,plan.shape,plan.surfaces,self.path.path,obstacles)
        self.durations=rest_to_rest_durations(p.pin,p.meta,self.path.path,self.path.distances)
        self.ends=np.r_[0.,np.cumsum(self.durations)]
        self.qs=[p.configuration(x)for x in self.path.path]
        self.inputs=[local.equilibrium(x)[0]for x in self.qs]
        self.flight=coupled.CoupledFlightController(p.pin,p.meta,qpin,track_vectoring=True,planned_input_reference=True)
        self.flight.command=np.where(local.dock,previous[:local.nj]/local.kp,previous[:local.nj])
        self.flight.force=previous[local.nj:].copy()
        self.entry=self.elapsed;self.ready=True;self.planning_s=time.perf_counter()-began
        self.info=dict(planning_s=self.planning_s,duration_s=float(self.ends[-1]),gap_m=gap,clearance=clearance)

    def update(self,q,v,previous):
        if not self.ready and self.elapsed>=.3-1e-9:self.prepare(q,v,previous)
        self.elapsed+=.01
        if not self.ready:return gravity_command(self.local,q,previous)[0]
        elapsed=self.elapsed-.01-self.entry;index=min(len(self.qs)-2,int(np.searchsorted(self.ends,elapsed,side='right')-1))
        target,tv,ta,s=quintic_reference(self.plant.pin,self.plant.meta,self.path.path[index],self.path.path[index+1],
            elapsed-self.ends[index],self.durations[index])
        u=(1-s)*self.inputs[index]+s*self.inputs[index+1]
        reference=self.plant.configuration(target);local=self.local
        bias=local.gain*u[:local.nj]-local.kp*reference[local.qi]
        self.query.qpos[:]=q;self.query.qvel[:]=v;qo,vo=self.plant.observation(self.query)
        cmd,thrust=self.flight.update(qo,vo,target,target_velocity=tv,target_acceleration=ta,joint_bias=bias,
            thrust_reference=u[local.nj:],gimbal_reference=target[self.flight.qi[self.flight.ai]])
        if self.flight.allocation_failure:raise ValueError(self.flight.allocation_failure)
        self.complete=elapsed>=self.ends[-1]
        return np.r_[np.where(local.dock,local.kp*cmd,cmd),thrust]
