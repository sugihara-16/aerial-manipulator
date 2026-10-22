"""Geometry/force-only preload refinement at an observed acquired posture."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import minimize
from build.load_saved_plan import load_saved_plan
from build.mit_drive import dock_mit_model
from build.sticking_design import LocalGrasp
from build.contact_step_control import pad_gap_jacobians
from build.check_local_pickup import prepare
from amsrr_det.mujoco_model import contact_dynamics_model

def refine(plant,q0,links,max_evaluations=160):
    m=plant.mj
    anchors=LocalGrasp(plant,q0,links);g0=anchors.contact()[0];pads=np.array([e[0] for e in anchors.elements])
    depth0=np.array([-g0[pads==i].min() for i in range(len(links))]);trace=[];best=None
    
    def shifted(depth):
        q=q0.copy();goal=g0-(depth-depth0)[pads]
        for _ in range(6):
            anchors.at(q);g,t,Jn,Ct=anchors.contact();residual=np.r_[goal-g,-t]
            if max(abs(residual))<1e-10:break
            J=np.vstack([Jn,Ct]);obj=m.jnt_dofadr[m.joint('object').id];J[:,obj:obj+6]=0
            change=np.linalg.lstsq(J,residual,rcond=1e-10)[0];mujoco.mj_integratePos(m,q,change,1.)
        return q
    
    def cost(depth_um):
        nonlocal best
        q=shifted(np.asarray(depth_um)*1e-6);local=LocalGrasp(plant,q,links)
        burden,u=local.static_burden(q)
        pressure=-local.k*local.contact()[0]
        loads=np.array([sum(pressure[k] for k,e in enumerate(local.elements) if e[0]==i)for i in range(len(links))])
        # Carry auditing requires every selected pad to exceed 0.5 N.
        # Include that same requirement in the common normalized usage cost.
        burden=max(burden,.5/loads.min() if loads.min()>0 else np.inf)
        trace.append(dict(depth_um=np.asarray(depth_um).tolist(),burden=float(burden),elements=len(local.elements),normal_loads=loads.tolist()))
        if best is None or burden<best[0]:best=(burden,q,local,np.asarray(depth_um))
        return min(burden,1e5)
    
    sol=minimize(cost,depth0*1e6,method='Powell',bounds=[(5.,500.)]*len(links),options=dict(maxfev=max_evaluations,xtol=.05,ftol=1e-4))
    burden,q,local,depth=best;local.at(q);real_depth=-pad_gap_jacobians(m,local.d,links)[0]
    info=dict(burden=burden,initial_depth_m=depth0.tolist(),desired_depth_m=real_depth.tolist(),
              evaluations=len(trace),optimizer_success=bool(sol.success),optimizer_message=str(sol.message))
    return q,local,info,trace

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False);start=time.perf_counter()
    plan,_=load_saved_plan('artifacts/runs/sticking_geometry_pair/candidate2')
    links=[f'm{a}/'+('yaw_dock_mech' if b.startswith('yaw') else 'pitch_dock_mech')+b[-1]for a,b in plan.surfaces]
    plant=contact_dynamics_model(dock_mit_model(plan.plant,kp=200.),links);m=plant.mj
    q0=np.load(args.state)['q'].copy()
    for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):q0[m.jnt_qposadr[j]+2]+=.15
    q,local,result,trace=refine(plant,q0,links)
    real_depth=np.array(result['desired_depth_m']);result['scope']=__doc__
    try:
        ground,supported,support_info,lift,carry,lower,upper=prepare(local,q,plan)
        result.update(support=support_info,lift=lift.info,carry=carry.info)
    except ValueError as error:result['preparation_error']=str(error)
    result['planning_s']=time.perf_counter()-start
    np.savez(out/'design.npz',q=q,normal_targets=real_depth)
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'search.json').write_text(json.dumps(trace,indent=2))
    for name in ['refine_captured_equilibrium.py','sticking_design.py','contact_step_control.py']:(out/name).write_bytes((Path('build')/name).read_bytes())
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
