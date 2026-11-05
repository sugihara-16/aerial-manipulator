"""Render the saved physical states; no controller or simulation rerun."""
import os
os.environ['MUJOCO_GL']='egl'
import json,sys,subprocess
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from lxml import etree as ET
root=Path(sys.argv[1]);names=sys.argv[2:]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
for name in names:
 p=root/name;doc=ET.parse(str(p/'physical.xml'));asset=doc.getroot().find('asset')
 # Visual ground references make stationary stance and forward swing visible.
 # Do not edit saved physical XML, geometry, friction, or simulation states.
 ET.SubElement(asset,'texture',name='audit_grid',type='2d',builtin='checker',width='512',height='512',rgb1='.64 .66 .69',rgb2='.8 .82 .85')
 ET.SubElement(asset,'material',name='audit_floor',texture='audit_grid',texrepeat='2 2',texuniform='true',reflectance='0')
 doc.find(".//geom[@name='floor']").set('material','audit_floor')
 m=mujoco.MjModel.from_xml_string(ET.tostring(doc,encoding='unicode'));d=mujoco.MjData(m)
 plan=json.loads((p/'selection.json').read_text())['plan'];states=json.loads((p/'states.json').read_text())
 feet={m.body(f'm{n}/'+('pitch'if port.startswith('pitch')else'yaw')+'_dock_mech'+port[-1]).id for n,port in plan['surfaces']}
 for i in range(m.ngeom):
  if m.geom(i).name=='floor':m.geom_rgba[i]=[.7,.72,.74,1]
  elif m.geom_bodyid[i]in feet:m.geom_rgba[i]=[.1,.7,.3,1]
 m.vis.global_.offwidth=800;m.vis.global_.offheight=600;m.vis.headlight.ambient[:]=.6
 cam=mujoco.MjvCamera();cam.distance=3.6 if plan['modules']<=4 else 4.2;cam.azimuth=110;cam.elevation=-18
 video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-s','800x600','-pix_fmt','rgb24','-r','10','-i','-','-threads','1','-c:v','libx264','-pix_fmt','yuv420p',str(p/'motion.mp4')],stdin=subprocess.PIPE)
 times=np.array([s['time_s']for s in states]);start=json.loads((p/'result.json').read_text())['whole_sequence']['walk_start_s'];marks=np.r_[.1,.8,start-7.5,start-3.,start-.1,start+.5,start+2.,start+7.,times[-1]];indices=set(np.argmin(abs(times[:,None]-marks),axis=0).tolist());frames=[]
 with mujoco.Renderer(m,height=600,width=800,max_geom=30000)as renderer:
  for index,s in enumerate(states):
   d.qpos[:]=s['q'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);cam.lookat[:]=d.subtree_com[0];cam.lookat[2]=max(.3,d.subtree_com[0,2]*.65)
   renderer.update_scene(d,cam);frame=renderer.render();im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,800,28),fill='white');draw.text((6,3),f'{name} | {s.get("phase","WALK")} | {s["time_s"]:.1f} s',fill='black',font=font);frame=np.asarray(im);video.stdin.write(frame.tobytes())
   if index in indices:
    im=Image.fromarray(frame).resize((480,360));draw=ImageDraw.Draw(im);draw.rectangle((0,0,480,25),fill='white');draw.text((4,1),f'{name} / {len(feet)} feet / {s["time_s"]:.1f}s',fill='black',font=font);frames.append(im)
 video.stdin.close()
 if video.wait()!=0:raise RuntimeError('video encoding failed')
 sheet=Image.new('RGB',(1440,1080),'white')
 for i,frame in enumerate(frames):sheet.paste(frame,((i%3)*480,(i//3)*360))
 sheet.save(p/'sequence_frames.jpg',quality=92)
 print(name,'rendered',flush=True)
