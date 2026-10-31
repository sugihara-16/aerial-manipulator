"""Render one recorded successful execution for each actual module count."""
import json,sys,math
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
root=Path(sys.argv[1]);chosen={}
qualified=None if not (root/'audit_summary.json').exists() else {r['id'] for r in json.loads((root/'audit_summary.json').read_text())['rows'] if r['success']}
for case in json.loads((root/'cases.json').read_text()):
 folder=root/case['id'];f=folder/'result.json'
 if not f.exists():continue
 result=json.loads(f.read_text())
 if result['task_success'] and (qualified is None or case['id'] in qualified):chosen.setdefault(result['modules'],(case,folder,result))
width,height=640,440;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
canvas=Image.new('RGB',(width*3,(height+56)*math.ceil(len(chosen)/3)),(245,245,245));draw=ImageDraw.Draw(canvas)
for i,(count,(case,folder,result)) in enumerate(sorted(chosen.items())):
 m=mujoco.MjModel.from_xml_path(str(folder/'physical.xml'));d=mujoco.MjData(m)
 states=np.load(folder/'physical_states.npz');trace=json.loads((folder/'physical_trace.json').read_text())
 # All views are observed LOAD-complete states, not a fabricated assembly.
 index=min(range(len(trace)),key=lambda k:abs(trace[k]['time_s']-5.))
 d.qpos[:]=states['q'][index];mujoco.mj_fwdPosition(m,d)
 ids=[b for b in range(m.nbody) if (m.body(b).name or '').startswith('m')]
 cloud=np.vstack([d.xpos[ids],np.array(case['request'].get('center',[0,0,1.5]))])
 lo,hi=cloud.min(axis=0)-.25,cloud.max(axis=0)+.25
 cam=mujoco.MjvCamera();cam.lookat[:]=(lo+hi)/2;cam.distance=max(2.,np.linalg.norm(hi-lo)*1.45);cam.azimuth=135;cam.elevation=-25
 option=mujoco.MjvOption();option.geomgroup[:]=1
 with mujoco.Renderer(m,height=height,width=width,max_geom=30000) as renderer:
  renderer.update_scene(d,camera=cam,scene_option=option);frame=Image.fromarray(renderer.render())
 x,y=(i%3)*width,(i//3)*(height+56);canvas.paste(frame,(x,y));draw.text((x+12,y+height+7),f'{count} modules / {result["contacts"]} contacts / {case["torque"]} Nm',fill=(20,20,20),font=font)
canvas.save(root/'structure_breadth.jpg',quality=92)
print(root/'structure_breadth.jpg')
