"""Physical gust comparison, both levels on the same CPU placement."""
import json,os,sys
from pathlib import Path
import numpy as np
import scripts.run_assembly_task as runner
from amsrr_det.assembly_air import AssemblyAir
from amsrr_det.assembly_fsm import AssemblyFSM
from amsrr_det.runtime_manifest import source_hashes
original_air=AssemblyAir.apply;original_fsm=AssemblyFSM.update
force=float(sys.argv[1]);protocol=dict(force_n=force,duration_s=.35,after_approach_s=.5,cpu_pair=[4,6],scope='physical gust; independent of main panel')
state=dict(start=None,force=None,body=None,last=None)
def observed_update(self,data):
    result=original_fsm(self,data)
    if state['start'] is None and self.stage=='APPROACH' and data.time-self.since>=protocol['after_approach_s']:
        edge=self.plan['rounds'][self.round]['edges'][0];direction=self.plan['directions'][edge]
        value=protocol['force_n']*np.array([-direction[1],direction[0],0.])
        state.update(start=float(data.time),force=value,body=self.m.body(f"m{self.edges[edge]['b']}/root").id)
    return result

def gust(self,data):
    if state['last'] is not None:data.xfrc_applied[state['body'],:3]-=state['last']
    original_air(self,data);state['last']=None
    if state['start'] is not None and state['start']<=data.time<state['start']+protocol['duration_s']:
        data.xfrc_applied[state['body'],:3]+=state['force'];state['last']=state['force'].copy()
AssemblyAir.apply=gust;AssemblyFSM.update=observed_update
os.sched_setaffinity(0,{4});case=next(c for c in json.load(open('configs/evaluation/assembly_generated.json'))['cases']if c['id']=='grasp/box_y0_0')
out=Path(f'artifacts/assembly_20261106/gust_{force:g}n_central27');r=runner.run(case,out,worker_cpu=6)
(out/'fault_protocol.json').write_text(json.dumps(dict(protocol,start_s=state['start'],direction=state['force'].tolist()),indent=2))
(out/'sources.json').write_text(json.dumps(source_hashes(),indent=2));print(json.dumps(r))
