"""Record this app on an isolated X display; never capture the user's desktop."""
import os,sys,json,time,subprocess
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from amsrr_det.desktop_gui import TaskWindow,STYLE
from amsrr_det.task_spec import defaults,make_spec
os.environ.pop('LD_PRELOAD',None)
subprocess.run(['sudo','-n','renice','-n','-15','-p',str(os.getpid())],check=True,stdout=subprocess.DEVNULL)
name=sys.argv[1];root=Path('artifacts/gui_direct_transition_20261111')/name;root.mkdir(parents=True,exist_ok=True)
kind=name.split('_')[0];values=defaults(kind)
if kind=='grasp':values.update(shape='ellipsoid',sx=.72,sy=1.29,sz=.70,mass=4.,dx=.4,dy=0.,height=.2,seed=1)
elif kind=='valve':values.update(torque=24.,tilt=0.,angle=0.,height=1.5,seed=1)
else:values.update(modules=6,speed=.5,distance=5.,curvature=0.,seed=1)
if len(sys.argv)>2:values.update(json.loads(sys.argv[2]))
spec=make_spec(kind,values,'integrated');(root/'requested_task.json').write_text(json.dumps(spec,indent=2))
f=QSurfaceFormat();f.setDepthBufferSize(24);f.setSwapInterval(0);f.setVersion(2,1);f.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile);QSurfaceFormat.setDefaultFormat(f)
os.environ.pop('LD_PRELOAD',None)
app=QApplication([]);app.setOverrideCursor(Qt.CursorShape.BlankCursor);app.setStyle('Fusion');app.setStyleSheet(STYLE)
w=TaskWindow(root/'runs');w.use_spec(spec);w.resize(1440,900);w.show();w.move(0,0)
video=root/'gui_full.mp4';log=(root/'capture.log').open('w')
recorder=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','x11grab','-draw_mouse','0','-framerate','60','-video_size','1440x900','-i',os.environ['DISPLAY'],'-c:v','h264_nvenc','-preset','p4','-cq','20','-b:v','0','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stdout=log,stderr=log)
began=time.monotonic();first_live=None;end=None;errors=[];frames=[];state_updates=[];telemetry=[];last_log=0.;abort_reason=None;last_progress=began;last_sim=-1.;airborne=False
def frame():
 frames.append(time.monotonic()-began)
 if not state_updates or state_updates[-1][1]!=w.viewer.sim_time:state_updates.append([frames[-1],w.viewer.sim_time])
w.viewer.frameSwapped.connect(frame);w.viewer.failed.connect(errors.append)
def tick():
 global first_live,end,last_log,last_sim,last_progress,abort_reason,airborne
 if w.loaded and w.viewer.model is not None:
  if first_live is None:first_live=time.monotonic()-began
  v=w.viewer
  if w.jobs.process is not None:os.sched_setaffinity(w.jobs.process.pid,{8})
  now=time.monotonic()
  if v.sim_time>last_sim+.001:last_progress=now;last_sim=v.sim_time
  positions=v.data.xpos[v.robot]
  # Smoothly frame all separated modules during assembly; tighten after joining.
  extent=float(__import__('numpy').ptp(positions,axis=0).max())
  distance=max(3.6,extent*1.55+2.5)
  v.camera.distance=.92*v.camera.distance+.08*distance
  v.camera.azimuth=125.;v.camera.elevation=-25.
  heights=positions[:,2]
  if float(heights.min())>.6:airborne=True
  if now-last_log>5:
   last_log=now
   row=dict(wall_s=now-began,simulation_s=v.sim_time,phase=v.phase,min_height=float(heights.min()),max_height=float(heights.max()),frames=len(frames))
   telemetry.append(row);print(json.dumps(row),flush=True)
   (root/'progress.json').write_text(json.dumps(row))
  reason=None
  if now-last_progress>15:reason='No new physical state for 15 seconds'
  if airborne and kind in ('valve','grasp') and v.phase not in ('TAKEOFF','OPEN','RETREAT') and float(heights.max())<.25:reason='All modules fell to the ground'
  transfer=w.jobs.path(w.current)/'transfer.json'
  if transfer.exists() and json.loads(transfer.read_text()).get('position_error_m',0)>.3:reason='Flight transfer missed its target by more than 30 cm'
  if (root/'stop.txt').exists():reason=(root/'stop.txt').read_text()
  if reason and not abort_reason and w.jobs.active:
   abort_reason=reason;print('STOP: '+reason,flush=True);w.stop_run()

 if w.current and w.jobs.active and not abort_reason:
  ready=w.jobs.path(w.current)/'ready.json'
  if ready.exists() and json.loads(ready.read_text())['preparation_s']>15:
   abort_reason='Initial preparation exceeded 15 seconds';print('STOP: '+abort_reason,flush=True);w.stop_run()
 if w.current and not w.jobs.active and end is None:end=time.monotonic()
 if end and time.monotonic()-end>3:
  state=w.jobs.state(w.current)
  selection=w.jobs.path(w.current)/'execution/selection.json'
  if kind=='grasp':selection=w.jobs.path(w.current)/'execution/generated_plan/plan.json'
  plan=json.loads(selection.read_text()) if selection.exists() else {}
  record=dict(task=spec,job=w.current,state=state,first_live_wall_s=first_live,wall_s=time.monotonic()-began,display_errors=errors,selection_path=str(selection))
  record.update(frame_times_s=frames,state_updates=state_updates,telemetry=telemetry,abort_reason=abort_reason,capture_fps=60)
  (root/'recording.json').write_text(json.dumps(record,indent=2))
  recorder.communicate(b'q',timeout=20);log.close();timer.stop();w.close();app.quit()
  print(json.dumps(dict(job=w.current,state=state.get('state'),success=(state.get('result') or {}).get('success'),reason=(state.get('result') or {}).get('reason'),abort_reason=abort_reason)),flush=True)
 if time.monotonic()-began>660:
  if w.jobs.active:w.jobs.cancel(w.jobs.active)
  recorder.communicate(b'q',timeout=20);app.exit(1)
timer=QTimer();timer.timeout.connect(tick);timer.start(150)
QTimer.singleShot(1000,lambda:QTest.mouseClick(w.start,Qt.MouseButton.LeftButton))
sys.exit(app.exec())
