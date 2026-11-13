import sys,time,json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from amsrr_det.desktop_gui import TaskWindow
app=QApplication([]);w=TaskWindow('artifacts/desktop_gui_20261111/cancel');w.show()
begin=time.monotonic()
def done():
 if w.current and not w.jobs.active and w.stopping is None:
  state=w.jobs.state(w.current);assert state['state']=='cancelled',state
  assert w.jobs.process is None
  Path('artifacts/desktop_gui_20261111/cancellation.json').write_text(json.dumps(state,indent=2))
  timer.stop();w.close();app.quit();print('Native Stop terminated the owned run and restored the controls',flush=True)
 elif time.monotonic()-begin>15:
  print('Cancellation timeout',flush=True);app.exit(1)
QTimer.singleShot(200,w.start_run);QTimer.singleShot(700,w.stop_run)
timer=QTimer();timer.timeout.connect(done);timer.start(100)
sys.exit(app.exec())
