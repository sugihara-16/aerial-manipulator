"""Saved physical states only: no trajectory or controller rerun."""
import os
os.environ['MUJOCO_GL']='egl'
import json,sys,subprocess
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from lxml import etree as ET
from amsrr_det.assembly_plan import groups
root=Path(sys.argv[1]);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
colors=np.array([[.2,.55,.8],[.9,.45,.15],[.25,.7,.4],[.65,.35,.8],[.85,.7,.2],[.25,.7,.75],[.8,.3,.5],[.55,.55,.55]])
for name in sys.argv[2:]:
 p=root/name;doc=ET.parse(str(p/'physical.xml'));asset=doc.getroot().find('asset')
 ET.SubElement(asset,'texture',name='assembly_grid',type='2d',builtin='checker',width='512',height='512',rgb1='.66 .68 .7',rgb2='.82 .84 .86')
 ET.SubElement(asset,'material',name='assembly_floor',texture='assembly_grid',texrepeat='2 2',texuniform='true',reflectance='0')
 doc.find(".//geom[@name='floor']").set('material','assembly_floor')
 m=mujoco.MjModel.from_xml_string(ET.tostring(doc,encoding='unicode'));d=mujoco.MjData(m);z=np.load(p/'trajectory.npz');result=json.load(open(p/'result.json'));n=result['modules'];clock=result['hover_confirmed_s'];edges=json.load(open(p/'input.json'))['edges']
 for i in range(m.ngeom):
  geom=m.geom(i).name or ''
  if geom.startswith('m'):
   module=int(geom.split('/')[0][1:]);m.geom_rgba[i,:3]=colors[module]
 m.vis.global_.offwidth=960;m.vis.global_.offheight=720;m.vis.headlight.ambient[:]=.6
 qi=np.array([m.jnt_qposadr[j]for j in range(m.njnt)if m.jnt_type[j]==0]);centres=z['q'][:,qi[:,None]+np.arange(3)]
 xy=centres[:,:,:2];center=(xy.min(axis=(0,1))+xy.max(axis=(0,1)))/2.;extent=max(np.ptp(xy[:,:,0]),np.ptp(xy[:,:,1]))
 cam=mujoco.MjvCamera();cam.distance=max(4.8,1.35*extent+2.0);cam.azimuth=110;cam.elevation=-37;cam.lookat[:]=[center[0],center[1],.8]
 video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-s','960x720','-pix_fmt','rgb24','-r','10','-i','-','-threads','1','-c:v','libx264','-pix_fmt','yuv420p',str(p/'assembly.mp4')],stdin=subprocess.PIPE)
 indices=set(np.linspace(0,len(z['t'])-1,6).astype(int));frames=[]
 with mujoco.Renderer(m,height=720,width=960,max_geom=30000)as renderer:
  for i,t in enumerate(z['t']):
   d.qpos[:]=z['q'][i];d.qvel[:]=z['v'][i];d.eq_active[:]=z['seams'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
   renderer.update_scene(d,cam);frame=renderer.render();im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,80),fill='white')
   draw.text((8,3),f'{name} | {n} modules | {z["phase"][i]} | {int(z["seams"][i].sum())}/{n-1} joins',fill='black',font=font)
   draw.text((8,28),f't={t:.1f}s | after hover: {max(0,t-clock):.1f}/30s | measured control delay + wake model',fill='black',font=font)
   membership=groups(n,edges,sum(1<<j for j,on in enumerate(z['seams'][i])if on))
   leaders=[min(j for j,g in enumerate(membership)if g==group)for group in sorted(set(membership))]
   draw.text((8,53),'Controller leaders: '+', '.join(map(str,leaders)),fill='black',font=font)
   video.stdin.write(np.asarray(im).tobytes())
   if i in indices:frames.append(im.resize((480,360)))
 video.stdin.close()
 if video.wait()!=0:raise RuntimeError('video encoding failed')
 sheet=Image.new('RGB',(1440,720),'white')
 for i,im in enumerate(frames):sheet.paste(im,((i%3)*480,(i//3)*360))
 sheet.save(p/'assembly_frames.jpg',quality=93);print(name,'rendered',flush=True)
