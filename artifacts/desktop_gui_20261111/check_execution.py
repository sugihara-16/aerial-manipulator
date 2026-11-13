import json,time,sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,Qt,QPointF
from PySide6.QtGui import QSurfaceFormat,QPainter,QMouseEvent
from amsrr_det.desktop_gui import TaskWindow,STYLE
from amsrr_det.task_spec import make_spec,defaults
out=Path('artifacts/desktop_gui_20261111');out.mkdir(exist_ok=True)
f=QSurfaceFormat();f.setDepthBufferSize(24);f.setVersion(2,1);f.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile);QSurfaceFormat.setDefaultFormat(f)
app=QApplication([]);app.setStyle('Fusion');app.setStyleSheet(STYLE)
w=TaskWindow(out/'runs');w.resize(1280,850);w.use_spec(make_spec('valve',defaults('valve')));w.show()
began=time.monotonic();captured=False;errors=[];w.viewer.failed.connect(errors.append)
def capture():
    frame=w.viewer.grabFramebuffer()
    assert not frame.isNull()
    frame.save(str(out/'viewport.png'))
    shot=w.grab();p=QPainter(shot)
    pos=w.viewer_container.mapTo(w,w.viewer_container.rect().topLeft())
    p.drawImage(pos,frame.scaled(w.viewer_container.size()));p.end();shot.save(str(out/'desktop.png'))
    before=w.viewer.camera.azimuth
    w.viewer.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress,QPointF(100,100),QPointF(100,100),Qt.MouseButton.LeftButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
    w.viewer.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove,QPointF(160,120),QPointF(160,120),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
    assert before!=w.viewer.camera.azimuth
    w.viewer.frame_all()
def tick():
    global captured
    try:
        if errors:raise RuntimeError(errors)
        if w.loaded and w.viewer.render_context is not None and not captured:
            capture();captured=True;print('Direct OpenGL frame and camera interaction verified',flush=True)
        if w.current and not w.jobs.active:
            state=w.jobs.state(w.current)
            assert captured,('No rendered frame',state)
            assert not (w.jobs.path(w.current)/'frame.jpg').exists()
            assert state['state']=='finished',state
            (out/'verification.json').write_text(json.dumps(dict(run=w.current,state=state,direct_opengl=True,camera_rotation=True,browser_renderer=False,display_errors=errors),indent=2))
            print('COMPLETE',state.get('result',{}).get('success'),flush=True)
            timer.stop();w.close();app.quit()
        if time.monotonic()-began>180:raise TimeoutError('UI integration check exceeded 180 s')
    except Exception as e:
        print('CHECK FAILED',repr(e),flush=True)
        if w.jobs.active:w.jobs.cancel(w.jobs.active);w.jobs.active=None
        timer.stop();w.close();app.exit(1)
timer=QTimer();timer.timeout.connect(tick);timer.start(200)
QTimer.singleShot(300,w.start_run)
sys.exit(app.exec())
