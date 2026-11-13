import json,sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,QPointF,Qt
from PySide6.QtGui import QSurfaceFormat,QPainter,QWheelEvent,QMouseEvent
from amsrr_det.desktop_gui import TaskWindow,STYLE
from amsrr_det.desktop_view import load_display_model
from amsrr_det.planning_preview import read_previews
out=Path('artifacts/desktop_gui_20261111')
folder=max((out/'runs').glob('*/live.json'),key=lambda p:p.stat().st_mtime).parent
f=QSurfaceFormat();f.setDepthBufferSize(24);f.setVersion(2,1);f.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile);QSurfaceFormat.setDefaultFormat(f)
app=QApplication([]);app.setStyle('Fusion');app.setStyleSheet(STYLE)
w=TaskWindow(out/'display_only');w.poll.stop();w.show();w.use_spec(json.loads((folder/'task.json').read_text()))
errors=[];w.viewer.failed.connect(errors.append)
def prepare():
 w.show_previews(read_previews(folder/'planning'))
 w.grab().save(str(out/'planning.png'))
 w.pages.setCurrentIndex(2);w.viewer.attach(folder,load_display_model(folder));w.viewer.selected=w.selected
 w.resize(1180,780)
def capture():
 try:
  assert not errors,errors
  frame=w.viewer.grabFramebuffer();frame.save(str(out/'viewport.png'))
  shot=w.grab();p=QPainter(shot);pos=w.viewer_container.mapTo(w,w.viewer_container.rect().topLeft());p.drawImage(pos,frame.scaled(w.viewer_container.size()));p.end();shot.save(str(out/'desktop.png'))
  before=w.viewer.camera.distance
  event=QWheelEvent(QPointF(100,100),QPointF(100,100),__import__('PySide6.QtCore',fromlist=['QPoint']).QPoint(0,0),__import__('PySide6.QtCore',fromlist=['QPoint']).QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
  w.viewer.wheelEvent(event);assert before!=w.viewer.camera.distance
  before=w.viewer.camera.lookat.copy()
  w.viewer.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress,QPointF(100,100),QPointF(100,100),Qt.MouseButton.RightButton,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier))
  w.viewer.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove,QPointF(160,120),QPointF(160,120),Qt.MouseButton.NoButton,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier))
  assert not w.viewer.follow and (before!=w.viewer.camera.lookat).any()
  w.viewer.release();w.viewer.attach(folder,load_display_model(folder))
  print('Resizing, zoom, pan, model reload and direct GL capture passed',flush=True)
  QTimer.singleShot(400,finish)
 except Exception as e:
  print(repr(e));w.close();app.exit(1)
def finish():
 assert not errors,errors
 w.close();app.quit()
QTimer.singleShot(300,prepare);QTimer.singleShot(1800,capture)
sys.exit(app.exec())
