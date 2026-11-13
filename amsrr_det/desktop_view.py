"""Qt-owned OpenGL display reading published states, never stepping physics."""
import json
import mmap
from pathlib import Path
import numpy as np
import mujoco
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QImage
from PySide6.QtOpenGL import QOpenGLWindow
from .display_scene import display_xml
from .task_live import snapshot


def draw_structure(painter, rect, row):
    """Neutral-joint top view using exactly the published connection tree."""
    flat = row.get('flat') if row else None
    if not flat:
        return
    poses = np.asarray(flat['poses'])
    centers = poses[:, :3, 3]
    points = list(centers) + [v for ports in flat['ports'] for v in ports.values()]
    points += [v for rotors in flat['rotors'] for v in rotors]
    xy = np.asarray(points)[:, [1, 0]] * [1, -1]
    low, high = xy.min(0)-.15, xy.max(0)+.15
    scale = float(min(rect.width()/max(.75, high[0]-low[0]), rect.height()/max(.75, high[1]-low[1])))
    def point(p):
        v = (np.asarray(p)[[1, 0]]*[1, -1]-(low+high)/2)*scale
        return QPointF(float(rect.center().x()+v[0]), float(rect.center().y()+v[1]))
    def line(a, b, color, width=2):
        painter.setPen(QPen(QColor(color), width));painter.drawLine(point(a), point(b))
    contacts = {(int(i), port) for i, port in row.get('surfaces', [])}
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for i, pose in enumerate(poses):
        for rotor in flat['rotors'][i]:
            line(centers[i], rotor, '#879dad')
            painter.setBrush(QColor('#33424f'));painter.drawEllipse(point(rotor), .09*scale, .09*scale)
        for port, tip in flat['ports'][i].items():
            line(centers[i], tip, '#e6ad64' if (i, port) in contacts else '#88a8bd', 3)
        body = [pose[:3, 3]+pose[:3, 0]*x+pose[:3, 1]*y for x, y in [(-.10,-.075),(.10,-.075),(.10,.075),(-.10,.075)]]
        painter.setBrush(QColor('#6487a4'));painter.setPen(QColor('#b2c7d7'))
        painter.drawPolygon(QPolygonF([point(p) for p in body]))
    for edge in row['edges']:
        line(flat['ports'][edge['a']][edge['port_a']], flat['ports'][edge['b']][edge['port_b']], '#c5d8ea', 3)
    labels = {}
    for i, center in enumerate(centers):
        key = tuple(np.round(center[:2], 4));labels.setdefault(key, [center, []])[1].append(str(i+1))
    for center, ids in labels.values():
        pos = point(center);text='/'.join(ids)
        painter.fillRect(QRectF(pos.x()-len(text)*4, pos.y()-8, len(text)*8, 16), QColor('#25292f'))
        painter.setPen(QColor('#ffffff'));painter.drawText(QRectF(pos.x()-40,pos.y()-9,80,18), Qt.AlignmentFlag.AlignCenter, text)


class SceneView(QOpenGLWindow):
    failed = Signal(str)
    def __init__(self):
        super().__init__()
        self.model = self.data = self.scene = self.render_context = None
        self.mapped = self.stream = None
        self.camera = mujoco.MjvCamera();self.option = mujoco.MjvOption()
        self.follow = True;self.selected = None;self.last_pos = None
        self.phase = '';self.sim_time = 0.;self.last_sequence = None
        self.robot = [];self.overlay=None;self.overlay_key=None;self.reset_camera()

    def attach(self, folder, model):
        self.release()
        self.model = model;self.data = mujoco.MjData(model)
        self.scene = mujoco.MjvScene(model, maxgeom=30000)
        self.robot = [i for i in range(model.nbody) if (model.body(i).name or '').startswith('m') and (model.body(i).name or '').endswith('/root')]
        self.stream = open(Path(folder)/'live.bin', 'rb')
        self.mapped = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.reset_camera();self.read_state();self.frame_all();self.requestUpdate()

    def release(self):
        if self.render_context:
            self.makeCurrent();self.render_context.free();self.render_context=None;self.doneCurrent()
        if self.mapped:self.mapped.close();self.mapped=None
        if self.stream:self.stream.close();self.stream=None
        self.model=self.data=self.scene=None
        self.phase='';self.sim_time=0.;self.last_sequence=None

    def reset_camera(self):
        self.camera.azimuth=120.;self.camera.elevation=-28.;self.camera.distance=4.
        self.camera.lookat[:]=[0,0,.8]

    def frame_all(self):
        if self.data is not None and self.robot:
            points=self.data.xpos[self.robot]
            self.camera.lookat[:]=points.mean(0)
            self.camera.distance=max(3.,float(np.ptp(points,axis=0).max())*1.4+2.)
        self.requestUpdate()

    def read_state(self):
        if self.mapped is None:return
        row=snapshot(self.mapped,{})
        if row is None:return
        t,phase,values=row
        if len(values)!=self.model.nq+self.model.nv+self.model.neq:
            raise ValueError('Published state does not match the display model')
        self.sim_time=t;self.phase=phase;self.data.time=t
        self.data.qpos[:]=values[:self.model.nq]
        self.data.qvel[:]=values[self.model.nq:self.model.nq+self.model.nv]
        self.data.eq_active[:]=values[self.model.nq+self.model.nv:]
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)
        if self.follow and self.robot:self.camera.lookat[:]=self.data.xpos[self.robot].mean(0)
        mujoco.mj_camlight(self.model,self.data)

    def paintGL(self):
        if self.model is None:return
        try:
            self.read_state()
            if self.render_context is None:
                self.render_context=mujoco.MjrContext(self.model,mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjv_updateScene(self.model,self.data,self.option,None,self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
            self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
            ratio=self.devicePixelRatio()
            rect=mujoco.MjrRect(0,0,int(self.width()*ratio),int(self.height()*ratio))
            painter=QPainter(self)
            painter.beginNativePainting()
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW,self.render_context)
            mujoco.mjr_render(rect,self.scene,self.render_context)
            # MuJoCo leaves depth/lighting state enabled. Qt's 2-D overlay
            # must not be tested against the rendered world's depth buffer.
            gl=self.context().functions()
            for flag in (0x0B71,0x0B44,0x0B50,0x0BC0,0x0C11):
                gl.glDisable(flag)  # depth, cull, lighting, alpha, scissor
            painter.endNativePainting()
            if self.selected:
                key=json.dumps(self.selected,sort_keys=True)
                if key!=self.overlay_key:
                    self.overlay_key=key
                    self.overlay=QImage(154,178,QImage.Format.Format_ARGB32_Premultiplied)
                    self.overlay.fill(QColor(30,34,41,240))
                    panel=QPainter(self.overlay)
                    panel.setPen(QColor('#cbd5e2'));panel.drawText(8,17,f"Selected · {self.selected['modules']} modules")
                    draw_structure(panel,QRectF(7,24,140,130),self.selected)
                    panel.setPen(QColor('#9ca9ba'));panel.drawText(8,169,'Top view · joints at 0°')
                    panel.end()
                painter.drawImage(10,10,self.overlay)
            painter.end()
        except Exception as error:
            self.failed.emit(str(error));self.release()

    def mousePressEvent(self,event):
        self.last_pos=event.position()

    def mouseMoveEvent(self,event):
        if self.scene is None or self.last_pos is None:return
        delta=event.position()-self.last_pos;self.last_pos=event.position()
        if event.buttons() & (Qt.MouseButton.RightButton|Qt.MouseButton.MiddleButton):
            action=mujoco.mjtMouse.mjMOUSE_MOVE_H;self.follow=False
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action=mujoco.mjtMouse.mjMOUSE_ROTATE_H
        else:return
        mujoco.mjv_moveCamera(self.model,action,delta.x()/max(1,self.height()),delta.y()/max(1,self.height()),self.camera)
        self.requestUpdate()

    def wheelEvent(self,event):
        if self.scene is not None:
            mujoco.mjv_moveCamera(self.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.08*event.angleDelta().y()/120,self.camera)
            self.requestUpdate()

    def mouseDoubleClickEvent(self,event):
        self.frame_all()


def load_display_model(folder):
    model=mujoco.MjModel.from_xml_string(display_xml((Path(folder)/'live.xml').read_text()))
    model.vis.headlight.ambient[:]=.18;model.vis.headlight.diffuse[:]=.35
    model.vis.quality.shadowsize=2048
    return model
