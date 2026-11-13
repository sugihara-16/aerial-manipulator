import time,os,json,sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import QTimer
from amsrr_det.desktop_gui import TaskWindow,STYLE
from amsrr_det.desktop_view import load_display_model
f=QSurfaceFormat();f.setDepthBufferSize(24);f.setVersion(2,1);f.setSwapInterval(0);f.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile);QSurfaceFormat.setDefaultFormat(f)
app=QApplication([]);app.setStyleSheet(STYLE);w=TaskWindow();w.resize(1440,900);w.show()
p=next(Path('artifacts/gui_recordings_v2_20261111/grasp/runs').iterdir());w.viewer.attach(p,load_display_model(p));w.pages.setCurrentWidget(w.viewer_container)
frames=[];w.viewer.frameSwapped.connect(lambda:frames.append(time.perf_counter()))
def done():
 import numpy as np
 dt=np.diff(frames);print(dict(fps=1/dt.mean(),p50_ms=float(np.median(dt)*1000),p99_ms=float(np.percentile(dt,99)*1000),count=len(frames)),flush=True);w.close();app.quit()
import cProfile,pstats
prof=cProfile.Profile();prof.enable();QTimer.singleShot(5000,done);app.exec();prof.disable();pstats.Stats(prof).sort_stats('cumtime').print_stats(18)
