import sys,numpy as np,mujoco
from pathlib import Path
from build.check_local_task import main
from build.load_saved_plan import load_saved_plan
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
plan,_=load_saved_plan(root/'generated_plan')
q=np.load(root/'endpoint.npz')['q'].copy()
for j in np.flatnonzero(plan.plant.mj.jnt_type==mujoco.mjtJoint.mjJNT_FREE):q[plan.plant.mj.jnt_qposadr[j]+2]+=.15
np.savez(out.parent/(out.name+'_air.npz'),q=q)
args=['--plan',str(root/'generated_plan'),'--goal',str(out.parent/(out.name+'_air.npz')),'--output',str(out),'--steps','80']
args+=['--'+x for x in 'predictive-motion distance-closing condensed-solve generated-feedback staged-approach hybrid-withdrawal momentum-feedback finite-slip-preload minimize-input governed-motion background-withdrawal bounded-allocation hold-withdrawal-pose preload-floor process-withdrawal observed-handoff inertial-input-updates nearby-handoff contact-coupled'.split()]
main(args+sys.argv[3:])
