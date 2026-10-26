"""Reduce input burden by a local gimbal direction step with physical rechecks.

The external-wrench optimization supplies a direction, not an accepted grasp.
Backtracking retains the finite-contact and task-feedback acceptance criteria.
This fixed procedure is part of each independent candidate evaluation.
"""
import time
import mujoco
from amsrr_det.clearance import MotionBoundedClearance
from amsrr_det.structure_bounds import hover_input_burden
from build.sticking_design import LocalGrasp
from build.task_gimbal_design import GimbalWrenchDesign
from build.finite_slip_preload import construct_finite_slip
from build.task_wrench_margin import refine_task_reference


def refine(plant,full_model,q,u,velocity,links,displacement,finite,wrench_info):
    began=time.perf_counter();local=LocalGrasp(plant,q,links)
    before=hover_input_burden(u[local.nj:]);records=[]
    unchanged=((q,u,velocity),finite,wrench_info)
    try:
        pose,design=GimbalWrenchDesign(local,q,displacement).solve(u,minimize_input=True)
        if design['accepted']:
            clearance=MotionBoundedClearance(full_model,links)
            for fraction in (1.,.5,.25):
                try:
                    candidate=q+fraction*(pose-q)  # only scalar gimbal coordinates change
                    (pq,pu,pv),pf=construct_finite_slip(plant,full_model,candidate,links,23.,
                        minimize_input=True,clearance=clearance)
                    (pq,pu,pv),pf,pw=refine_task_reference(plant,full_model,pq,pu,pv,links,displacement,pf)
                    cost=hover_input_burden(pu[local.nj:])
                    records.append(dict(fraction=fraction,burden=cost))
                    if cost<before-1e-7:
                        return ((pq,pu,pv),pf,pw,dict(before=before,after=cost,
                            records=records,wall_s=time.perf_counter()-began))
                except (ValueError,mujoco.FatalError) as error:
                    records.append(dict(fraction=fraction,reason=str(error)))
    except (ValueError,mujoco.FatalError) as error:
        records.append(dict(reason=str(error)))
    return (*unchanged,dict(before=before,after=before,records=records,wall_s=time.perf_counter()-began))
