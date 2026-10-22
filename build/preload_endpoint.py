"""Construct a static endpoint; recheck actual geometry after each local step.

Acceptance means force-balance existence and conservative physical bounds at
one pose. It says nothing about reaching that pose or completing a task.
"""
import numpy as np
import mujoco
from build.sticking_design import LocalGrasp
from build.linear_preload import preload_step
from build.contact_step_control import pad_gap_jacobians
from amsrr_det.clearance import MotionBoundedClearance


def construct(plant, full_model, q, links, iterations=12):
    clearance=MotionBoundedClearance(full_model,links)
    records=[]
    for iteration in range(iterations):
        local=LocalGrasp(plant,q,links)
        if len(set(e[0]for e in local.elements))<len(links):
            # Contact-free touching poses have no force elements. Seed the
            # active set by 1 micrometre of geometric closure, then recompute
            # exact finite geometry. This is only a numerical initial pose;
            # the force equations below choose the actual required preload.
            for _ in range(6):
                local.at(q);gaps,J=pad_gap_jacobians(plant.mj,local.d,links)
                obj=plant.mj.jnt_dofadr[plant.mj.joint('object').id];J[:,obj:obj+6]=0.
                J[:,local.vi[~local.dock]]=0.
                delta=np.linalg.lstsq(J,np.minimum(-1e-6-gaps,0.),rcond=1e-10)[0]
                if not np.isfinite(delta).all() or abs(delta).max()>.005:
                    return None,dict(reason='contact_active_set_seed_unresolved',iterations=records)
                q=q.copy();mujoco.mj_integratePos(plant.mj,q,delta,1.)
                local=LocalGrasp(plant,q,links)
                if len(set(e[0]for e in local.elements))==len(links):break
            else:return None,dict(reason='contact_active_set_seed_unresolved',iterations=records)
        try:
            target,step=preload_step(local,q,fixed_material=False)
        except ValueError as error:
            return None,dict(reason=str(error),iterations=records)
        actual=LocalGrasp(plant,target,links)
        burden,u=actual.static_burden(target)
        actual.at(target);gap=actual.contact()[0]
        normal=np.array([sum(-actual.k*gap[i]for i,e in enumerate(actual.elements)if e[0]==j)for j in range(len(links))])
        # Triangle inequalities cover EVERY Coulomb-admissible force split.
        factor=np.sqrt(1+4.5**2)
        moment=np.array([sum(factor*(-actual.k*gap[i])*np.linalg.norm(e[3])
                        for i,e in enumerate(actual.elements)if e[0]==j)for j in range(len(links))])
        bad,_,_=clearance.check(target,True)
        commands_ok=False
        if u is not None:
            limited=plant.mj.actuator_ctrllimited.astype(bool)
            commands_ok=bool(np.all(u[limited]>=plant.mj.actuator_ctrlrange[limited,0]) and
                             np.all(u[limited]<=plant.mj.actuator_ctrlrange[limited,1]))
            commands_ok &= bool(np.all(u[:actual.nj][~actual.dock]>=plant.mj.jnt_range[actual.joint_ids[~actual.dock],0]) and
                                np.all(u[:actual.nj][~actual.dock]<=plant.mj.jnt_range[actual.joint_ids[~actual.dock],1]))
        feasible=bool(burden<1 and len(gap)>0 and np.all(gap>=-.002) and
            np.all(normal>=.5) and np.all(factor*normal<=30.) and np.all(moment<=5.) and not bad and commands_ok and
            np.all(target[actual.qi]>=plant.mj.jnt_range[actual.joint_ids,0]) and
            np.all(target[actual.qi]<=plant.mj.jnt_range[actual.joint_ids,1]))
        records.append(dict(iteration=iteration,static_burden=burden,normal_n=normal.tolist(),
            conservative_force_n=(factor*normal).tolist(),conservative_moment_nm=moment.tolist(),
            geometry_violation=bool(bad),command_bounds_ok=commands_ok,actual_endpoint_feasible=feasible,
            local_predicted_normal_n=step['predicted_loads'],contact_elements=len(actual.elements)))
        if feasible:return (target,u),dict(reason='static_endpoint',iterations=records)
        q=target
    return None,dict(reason='local_iteration_limit',iterations=records)
