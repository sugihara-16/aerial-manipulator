"""Predeclared subset; same fixed input and CPU placement, finer physics step."""
import json,os,subprocess
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes
os.sched_setaffinity(0,{31})
root=Path('artifacts/assembly_20261106/full219_central27');out=root.parent/'halfstep6_central27';out.mkdir(exist_ok=True)
ids=['valve/low_tilt_down_t0.75_a0','grasp/box_y0_0','grasp/cylinder_y1_1','grasp/box_y3_0','valve/high_down_t60_a1','walking/n8_v30_straight']
(out/'protocol.json').write_text(json.dumps(dict(cases=ids,timestep_s=.00025,air_update_s=.002,scope='same input/seed and CPU pairs; sequential sensitivity run, measured latency sampled anew'),indent=2))
if __name__=='__main__':
 source={r['case']:r for r in json.load(open(root/'summary.json'))['results']};rows=[];hashes=json.load(open(root/'sources.json'))
 for case in ids:
  if source_hashes()!=hashes:raise RuntimeError('runtime changed')
  cpus=source[case]['cpu_pair'];folder=out/case;folder.parent.mkdir(parents=True,exist_ok=True)
  with (folder.parent/(folder.name+'.log')).open('w')as log:
   subprocess.run(['.venv/bin/python','-m','scripts.run_assembly_task',case,str(folder),'--timestep','.00025','--control-cpu',str(cpus[0]),'--worker-cpu',str(cpus[1])],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
  r=json.load(open(folder/'result.json'));r.update(original_success=source[case]['success'],cpu_pair=cpus);rows.append(r)
  (out/'summary.json').write_text(json.dumps(rows,indent=2));print(case,r['success'],r['reason'],flush=True)
