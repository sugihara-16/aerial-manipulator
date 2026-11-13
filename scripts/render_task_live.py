"""Read-only MuJoCo display process. Never drives or delays the controller."""
import os
os.environ['MUJOCO_GL']='egl'
import argparse,json,mmap,time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from amsrr_det.task_live import snapshot
from amsrr_det.task_execution import write


from amsrr_det.display_scene import display_xml


def render(folder,frames=None):
    p=Path(folder);meta=json.loads((p/'live.json').read_text());m=mujoco.MjModel.from_xml_string(display_xml((p/'live.xml').read_text()));d=mujoco.MjData(m)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=640;m.vis.headlight.ambient[:]=.18
    m.vis.headlight.diffuse[:]=.35;m.vis.quality.shadowsize=4096
    robot=[i for i in range(m.nbody) if (m.body(i).name or '').startswith('m') and (m.body(i).name or '').endswith('/root')]
    sun=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_LIGHT,'console_sun')
    camera=mujoco.MjvCamera();camera.azimuth=120.;camera.elevation=-28.;camera.distance=5.
    with open(p/'live.bin','rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm,mujoco.Renderer(m,height=640,width=960,max_geom=30000) as renderer:
        started=time.monotonic();count=0
        while frames is None or count<frames:
            tick=time.monotonic()
            row=snapshot(mm,meta)
            if row:
                t,phase,values=row;d.qpos[:]=values[:m.nq];d.qvel[:]=values[m.nq:m.nq+m.nv];d.eq_active[:]=values[m.nq+m.nv:]
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                positions=d.xpos[robot];center=positions.mean(axis=0)
                camera.lookat[:]=center;camera.distance=max(3.,float(np.ptp(positions,axis=0).max())*1.4+2.)
                m.light_pos[sun]=center+[0.,0.,8.]
                m.vis.map.shadowclip=max(1.,camera.distance/m.stat.extent)
                mujoco.mj_camlight(m,d)
                renderer.update_scene(d,camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
                frame=renderer.render()
                temp=p/'frame.tmp.jpg';Image.fromarray(frame).save(temp,quality=85);temp.replace(p/'frame.jpg')
                count+=1
                write(p/'live_status.json',dict(time_s=t,phase=phase,render_fps=(count-1)/max(time.monotonic()-started,1e-6)))
            time.sleep(max(0.,1/30-(time.monotonic()-tick)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder');ap.add_argument('--frames',type=int);a=ap.parse_args()
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[-1]})
    render(a.folder,a.frames)
