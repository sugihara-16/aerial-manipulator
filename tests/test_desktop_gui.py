"""Native controls retain the existing input contract and execution ownership."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from types import SimpleNamespace
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from amsrr_det.desktop_gui import TaskWindow
from amsrr_det.task_spec import defaults,make_spec

@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])

@pytest.fixture
def window(tmp_path,app):
    w=TaskWindow(tmp_path);w.poll.stop();w.draw_timer.stop()
    yield w
    w.close();w.deleteLater()
    from PySide6.QtCore import QCoreApplication,QEvent
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    app.processEvents()

@pytest.mark.parametrize('kind',['grasp','valve','walking'])
@pytest.mark.parametrize('mode',['task','integrated','assembly'])
def test_native_form_round_trip(window,kind,mode):
    spec=make_spec(kind,defaults(kind),mode)
    window.use_spec(spec)
    assert window.specification()==spec
    assert json.loads(window.spec_text.toPlainText())==spec


def test_invalid_json_does_not_replace_current_form(window):
    before=window.specification()
    window.spec_text.setPlainText('{broken')
    window.apply_spec()
    assert window.specification()==before
    assert window.error.text()


def test_start_uses_shared_contract_without_browser_renderer(window,monkeypatch):
    captured=[]
    def start(spec):captured.append(spec);return 'a'*32
    monkeypatch.setattr(window.jobs,'start',start)
    monkeypatch.setattr(window,'refresh',lambda:None)
    window.start_run()
    assert captured==[window.specification()]
    assert window.jobs.render_frames is False


def test_monitor_without_jpeg_renderer_does_not_time_out_live_run(tmp_path,monkeypatch):
    from amsrr_det.task_server import Jobs
    from amsrr_det.task_execution import write
    import amsrr_det.task_server as server
    key='a'*32;folder=tmp_path/key;folder.mkdir()
    (folder/'live.json').write_text('{}')
    write(folder/'status.json',dict(state='finished',result={'success':True}))
    times=iter([0.,50.]);monkeypatch.setattr(server.time,'monotonic',lambda:next(times))
    monkeypatch.setattr(server.time,'sleep',lambda _:None)
    def no_renderer(*args,**kwargs):raise AssertionError('No renderer process expected')
    monkeypatch.setattr(server.subprocess,'Popen',no_renderer)
    polls=iter([None,0]);process=SimpleNamespace(poll=lambda:next(polls),wait=lambda:0)
    jobs=Jobs(tmp_path,render_frames=False)
    monkeypatch.setattr(jobs,'kill',lambda _:pytest.fail('A live native run must not hit the 45 s initialization limit'))
    with (folder/'log').open('w') as log:jobs.monitor(key,process,log,{})
    assert json.loads((folder/'status.json').read_text())['state']=='finished'


def test_structure_preview_renders_numeric_geometry(app):
    import numpy as np
    from PySide6.QtGui import QImage,QPainter,QColor
    from PySide6.QtCore import QRectF
    from amsrr_det.desktop_view import draw_structure
    row=dict(modules=1,edges=[],surfaces=[[0,'yaw1']],flat=dict(
        poses=[np.eye(4).tolist()],ports=[{'yaw1':[.2,0,0]}],
        rotors=[[[.1,.1,0],[-.1,.1,0],[.1,-.1,0],[-.1,-.1,0]]]))
    image=QImage(240,240,QImage.Format.Format_ARGB32);image.fill(QColor('black'))
    painter=QPainter(image)
    try:draw_structure(painter,QRectF(0,0,240,240),row)
    finally:painter.end()
    assert image.pixelColor(120,120)!=QColor('black')


def test_display_batching_keeps_triangles_and_physical_input(tmp_path):
    import xml.etree.ElementTree as ET
    import numpy as np
    from amsrr_det.display_scene import display_xml
    mesh=tmp_path/'part.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    xml=f'<mujoco><asset><mesh name="a" file="{mesh}"/><mesh name="b" file="{mesh}" scale="2 2 2"/></asset><worldbody><body><joint name="hinge"/><geom name="m0/a/0" type="mesh" mesh="a"/><geom name="m0/a/1" type="mesh" mesh="b"/></body></worldbody></mujoco>'
    shown=ET.fromstring(display_xml(xml));original=ET.fromstring(xml)
    assert len(shown.findall('.//body/geom'))==1
    assert len(original.findall('.//body/geom'))==2
    joined=shown.find('asset/mesh')
    np.testing.assert_array_equal(np.fromstring(joined.get('face'),sep=' ',dtype=int),[0,1,2,3,4,5])
    np.testing.assert_array_equal(np.fromstring(joined.get('vertex'),sep=' ').reshape(-1,3),[[0,0,0],[1,0,0],[0,1,0],[0,0,0],[2,0,0],[0,2,0]])
    assert shown.find('.//joint').attrib==original.find('.//joint').attrib
