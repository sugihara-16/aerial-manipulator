"""Independent minimum-time check over the recorded feasible transition graph."""
import ast,json,sys
from pathlib import Path
from functools import lru_cache
import numpy as np,pinocchio as pin
from amsrr_det.assembly import make_assembly_plant
from amsrr_det.assembly_plan import groups
from amsrr_det.model import PORTS
@lru_cache(None)
def audit_plant(n):
    # This audit uses only free-module joint coordinates and kinematics.
    # Connection constraints and collision geometry are not evaluated here.
    return make_assembly_plant(n,[],meshes=False)

root=Path(sys.argv[1]);panel=json.load(open(root/'panel.json'))['cases'];rows=[]
for c in panel:
 folder=root/c['id']
 if not (folder/'result.json').exists() or not (folder/'plan.json').exists():continue
 plan=json.load(open(folder/'plan.json'));saved=np.load(folder/'planned.npz');p=audit_plant(c['modules']);m=p.mj
 import mujoco
 d=mujoco.MjData(m);d.qpos[:]=saved['closed'];q,_=p.observation(d);pd=p.pin.createData();pin.framesForwardKinematics(p.pin,pd,q)
 qi=[m.jnt_qposadr[m.joint(p.pin.names[b['joint']]).id]for b in p.meta['bases']];edges=c['edges'];n=c['modules'];full=(1<<len(edges))-1
 offsets=[]
 for e in edges:
  nodes={e['b']}
  for _ in range(n):nodes|={x['b']for x in edges if x['a']in nodes}
  axis=pd.oMf[p.meta['frames'][f"m{e['a']}/{PORTS[e['port_a']]}"]].rotation[:,0]
  offset=np.zeros((n,3));offset[list(nodes)]=.18*axis;offsets.append(offset)
 def position(mask):return np.array([saved['closed'][k:k+3]for k in qi])+sum((x for i,x in enumerate(offsets)if not mask&(1<<i)),start=np.zeros((n,3)))
 checked={ast.literal_eval(k):v for k,v in plan['screened_paths'].items()};seen={a for a,b in checked}|{full};missing=0;invalid_parallel=0
 for mask in seen-{full}:
  comp=groups(n,edges,mask);remaining=[i for i in range(len(edges))if not mask&(1<<i)]
  for bits in range(1,1<<len(remaining)):
   chosen=[remaining[j]for j in range(len(remaining))if bits&(1<<j)];used=[comp[edges[i][key]]for i in chosen for key in ('a','b')]
   if len(used)!=len(set(used)):continue
   next_mask=mask|sum(1<<i for i in chosen)
   missing+=(mask,next_mask) not in checked
 def cost(a,b):
  distance=float(np.linalg.norm(position(b)-position(a),axis=1).max())
  return max(1.5,1.875*distance/.18,np.sqrt(5.78*distance/.5))+plan.get('merge_overhead_s',1.35)
 @lru_cache(None)
 def best(mask,serial):
  if mask==full:return 0.
  options=[cost(a,b)+best(b,serial)for(a,b),valid in checked.items()if a==mask and valid and (not serial or (b^a).bit_count()==1)]
  return min(options,default=float('inf'))
 selected=sum(cost(r['start_mask'],r['end_mask'])for r in plan['rounds'])
 for r in plan['rounds']:
  comp=groups(n,edges,r['start_mask']);used=[comp[edges[i][key]]for i in r['edges']for key in ('a','b')]
  invalid_parallel+=len(used)!=len(set(used)) or not checked.get((r['start_mask'],r['end_mask']),False)
 good=missing==0 and invalid_parallel==0 and abs(selected-best(0,False))<1e-7 and abs(selected-plan['predicted_merge_s'])<1e-7
 rows.append(dict(case=c['id'],ok=good,checked=len(checked),missing=missing,parallel=any(len(r['edges'])>1 for r in plan['rounds']),selected_s=selected,best_s=best(0,False),serial_s=best(0,True)))
result=dict(count=len(rows),issues=[r for r in rows if not r['ok']],parallel=sum(r['parallel']for r in rows),rows=rows)
(root/'schedule_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))
