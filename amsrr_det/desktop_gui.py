"""Native task console. Qt controls and MuJoCo rendering, without an HTTP server."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QAction, QColor, QPainter, QSurfaceFormat, QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox, QLabel,
    QPushButton, QDockWidget, QScrollArea, QStackedWidget, QPlainTextEdit,
    QFileDialog, QMessageBox, QGridLayout, QToolBar)
from .task_spec import CATALOG, COMMON, make_spec, validate_spec, form_values
from .task_server import Jobs, ROOT
from .planning_preview import read_previews
from .desktop_view import SceneView, draw_structure, load_display_model

STYLE = '''
QMainWindow,QWidget{background:#27292e;color:#cbd2dd;font-size:12px;}
QToolBar{background:#303239;border:0;border-bottom:1px solid #15171b;spacing:7px;padding:5px;}
QDockWidget{font-weight:600;} QDockWidget::title{background:#32353c;padding:7px;}
QLineEdit,QAbstractSpinBox,QComboBox,QPlainTextEdit{background:#1e2025;border:1px solid #4b505a;border-radius:2px;padding:5px;selection-background-color:#416b9d;}
QPushButton{background:#383c44;border:1px solid #4b505a;border-radius:3px;padding:6px 12px;}
QPushButton:hover{background:#474e59;} QPushButton:disabled{color:#737780;}
QPushButton#start{background:#416b9d;color:white;} QPushButton#start:disabled{background:#343e4c;color:#8593a5;} QLabel#hint{color:#98a4b5;}
QScrollArea{border:0;} QStatusBar{background:#1c1e23;} QToolTip{background:#30343c;color:white;border:1px solid #697383;}
'''


class StructureCard(QWidget):
    def __init__(self,row,parent=None):
        super().__init__(parent);self.row=row;self.setMinimumSize(180,230)
    def paintEvent(self,event):
        p=QPainter(self);row=self.row
        p.fillRect(self.rect(),QColor('#282d35'))
        p.setPen(QColor('#79b49a' if row.get('selected') else '#485465'))
        p.drawRect(self.rect().adjusted(0,0,-1,-1))
        p.setPen(QColor('#d7deea'));p.drawText(10,22,f"{row['modules']} modules")
        view=row.get('validated') or row
        draw_structure(p,QRectF(10,30,self.width()-20,self.height()-82),view)
        c=row.get('checks',{})
        text='Selected' if row.get('selected') else 'Model checks passed' if row.get('validated') else 'Growing · not validated'
        p.setPen(QColor('#a2b0c2'));p.drawText(10,self.height()-28,text)
        p.drawText(10,self.height()-11,f"Checked {c.get('checked',0)}   Passed {c.get('accepted',0)}")
        p.end()


class TaskWindow(QMainWindow):
    def __init__(self,folder=None):
        super().__init__();self.setWindowTitle('A-MSRR — Task Workspace');self.resize(1280,850)
        self.jobs=Jobs(folder or ROOT/'artifacts/desktop_console',render_frames=False)
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='display')
        self.current=None;self.widgets={};self.preview_future=None;self.model_future=None
        self.loaded=False;self.load_failed=False;self.preview_stamp=None;self.result_stamp=None
        self.stopping=None;self.closed=False;self.selected=None
        bar=QToolBar('Execution',self);bar.setMovable(False);self.addToolBar(bar)
        self.start=QPushButton('▶ Start');self.start.setObjectName('start');self.start.clicked.connect(self.start_run)
        self.stop=QPushButton('■ Stop');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_run)
        bar.addWidget(self.start);bar.addWidget(self.stop);bar.addSeparator()
        for label,callback in [('Load task',self.load_task),('Save task',self.save_task),('Fit view',self.fit_view)]:
            action=QAction(label,self);action.triggered.connect(callback);bar.addAction(action)
        self.follow=QAction('Follow robot',self);self.follow.setCheckable(True);self.follow.setChecked(True)
        self.follow.triggered.connect(self.set_follow);bar.addAction(self.follow)
        self.run_state=QLabel('  Idle');bar.addWidget(self.run_state)
        self.viewer=SceneView();self.viewer.failed.connect(self.display_error)
        self.viewer_container=QWidget.createWindowContainer(self.viewer)
        self.viewer_container.setMinimumSize(300,240)
        self.pages=QStackedWidget();self.setCentralWidget(self.pages)
        idle=QLabel('Ready to simulate\n\nConfigure the task, then select Start.');idle.setAlignment(Qt.AlignmentFlag.AlignCenter);idle.setObjectName('hint')
        self.pages.addWidget(idle)
        self.preview_scroll=QScrollArea();self.preview_scroll.setWidgetResizable(True)
        self.preview_body=QWidget();self.preview_layout=QGridLayout(self.preview_body)
        self.preview_scroll.setWidget(self.preview_body);self.pages.addWidget(self.preview_scroll)
        self.pages.addWidget(self.viewer_container)
        self.properties=QWidget();layout=QVBoxLayout(self.properties)
        form=QFormLayout();self.kind=QComboBox();self.mode=QComboBox()
        for key,value in CATALOG.items():self.kind.addItem(value['label'],key)
        for key,label in [('task','Assembled start'),('integrated','Take off, assemble, run task'),('assembly','Assembly only')]:self.mode.addItem(label,key)
        form.addRow('Task',self.kind);form.addRow('Execution',self.mode);layout.addLayout(form)
        self.scope=QLabel();self.scope.setWordWrap(True);self.scope.setObjectName('hint');layout.addWidget(self.scope)
        self.fields=QFormLayout();layout.addLayout(self.fields)
        self.error=QLabel();self.error.setWordWrap(True);self.error.setStyleSheet('color:#f3a49c');layout.addWidget(self.error)
        self.spec_toggle=QPushButton('Task specification');layout.addWidget(self.spec_toggle)
        self.spec_text=QPlainTextEdit();self.spec_text.setMinimumHeight(180);self.spec_text.hide();layout.addWidget(self.spec_text)
        self.apply=QPushButton('Apply specification');self.apply.hide();self.apply.clicked.connect(self.apply_spec);layout.addWidget(self.apply)
        self.spec_toggle.clicked.connect(self.toggle_spec);layout.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.properties)
        dock=QDockWidget('Task properties',self);dock.setWidget(scroll);dock.setMinimumWidth(310)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable|QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,dock)
        results=QWidget();result_layout=QVBoxLayout(results)
        self.outcome=QLabel('No execution results.');self.outcome.setWordWrap(True);result_layout.addWidget(self.outcome)
        self.metrics=QLabel('Modules —     Planning / setup —     Max. command time —');result_layout.addWidget(self.metrics)
        row=QHBoxLayout();details=QPushButton('Result details');row.addWidget(details)
        self.open_files=QPushButton('Open run folder');self.open_files.setEnabled(False);self.open_files.clicked.connect(self.open_folder);row.addWidget(self.open_files);row.addStretch();result_layout.addLayout(row)
        self.detail=QPlainTextEdit();self.detail.setReadOnly(True);self.detail.hide();result_layout.addWidget(self.detail)
        details.clicked.connect(lambda:self.detail.setVisible(not self.detail.isVisible()))
        output=QDockWidget('Execution results',self);output.setWidget(results)
        output.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable|QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,output)
        self.statusBar().showMessage('Ready · Left drag: rotate · Right drag: pan · Wheel: zoom · Double-click: fit')
        self.kind.currentIndexChanged.connect(self.make_fields);self.mode.currentIndexChanged.connect(self.update_spec)
        self.make_fields()
        self.poll=QTimer(self);self.poll.timeout.connect(self.refresh);self.poll.start(150)
        self.draw_timer=QTimer(self);self.draw_timer.timeout.connect(self.viewer.requestUpdate);self.draw_timer.start(16)

    def make_fields(self,*args,values=None):
        while self.fields.rowCount():self.fields.removeRow(0)
        self.widgets={}
        for field in CATALOG[self.kind.currentData()]['fields']+COMMON:
            if field['choices']:
                editor=QComboBox()
                for item in field['choices']:editor.addItem(item.replace('_',' ').title(),item)
                editor.setCurrentIndex(editor.findData((values or {}).get(field['key'],field['default'])))
                editor.currentIndexChanged.connect(self.update_spec)
            else:
                editor=QSpinBox() if field['key'] in ('modules','seed') else QDoubleSpinBox()
                if isinstance(editor,QDoubleSpinBox):editor.setDecimals(6)
                editor.setRange(field['min'],field['max']);editor.setSingleStep(field.get('step') or .01)
                editor.setValue((values or {}).get(field['key'],field['default']))
                editor.valueChanged.connect(self.update_spec)
                editor.setKeyboardTracking(False)
            editor.setObjectName(field['key']);self.widgets[field['key']]=editor
            label=QLabel(field['label']);label.setWordWrap(True);label.setBuddy(editor)
            self.fields.addRow(label,editor)
        self.update_spec()

    def specification(self):
        values={k:(w.currentData() if isinstance(w,QComboBox) else w.value()) for k,w in self.widgets.items()}
        return make_spec(self.kind.currentData(),values,self.mode.currentData())

    def update_spec(self,*args):
        self.scope.setText({'task':'Spawn the generated structure in the air and execute the task.',
            'assembly':'Take off, assemble, and verify stable hovering.',
            'integrated':'Take off and assemble, then continue into the task without resetting physics.'}[self.mode.currentData()])
        try:self.spec_text.setPlainText(json.dumps(self.specification(),indent=2));self.error.clear()
        except ValueError as error:self.error.setText(str(error))

    def toggle_spec(self):
        visible=not self.spec_text.isVisible();self.spec_text.setVisible(visible);self.apply.setVisible(visible)

    def use_spec(self,spec):
        spec=validate_spec(spec)
        self.kind.blockSignals(True);self.mode.blockSignals(True)
        self.kind.setCurrentIndex(self.kind.findData(spec['task']))
        self.mode.setCurrentIndex(self.mode.findData(spec['execution']['mode']))
        self.kind.blockSignals(False);self.mode.blockSignals(False)
        self.make_fields(values=form_values(spec))

    def apply_spec(self):
        try:self.use_spec(json.loads(self.spec_text.toPlainText()))
        except (ValueError,TypeError) as error:self.error.setText(str(error))

    def load_task(self):
        name,_=QFileDialog.getOpenFileName(self,'Load task',str(ROOT/'configs/tasks'),'JSON (*.json)')
        if name:
            try:self.use_spec(json.loads(Path(name).read_text()))
            except (OSError,ValueError) as error:self.error.setText(str(error))

    def save_task(self):
        try:spec=self.specification()
        except ValueError as error:self.error.setText(str(error));return
        name,_=QFileDialog.getSaveFileName(self,'Save task','task.json','JSON (*.json)')
        if name:
            try:Path(name).write_text(json.dumps(spec,indent=2)+'\n')
            except OSError as error:self.error.setText(str(error))

    def start_run(self):
        try:
            spec=self.specification();key=self.jobs.start(spec)
        except (ValueError,OSError) as error:self.error.setText(str(error));return
        self.current=key;self.loaded=False;self.load_failed=False;self.selected=None
        self.preview_stamp=None;self.result_stamp=None;self.viewer.release();self.viewer.selected=None
        self.outcome.setText('Running');self.metrics.setText('Modules —     Planning / setup —     Max. command time —');self.detail.clear()
        self.pages.setCurrentIndex(0);self.open_files.setEnabled(True);self.error.clear();self.refresh()

    def stop_run(self):
        if self.jobs.active and self.stopping is None:
            self.stopping=self.pool.submit(self.jobs.cancel,self.jobs.active);self.stop.setEnabled(False)

    def fit_view(self):self.viewer.frame_all()
    def set_follow(self,on):self.viewer.follow=on
    def open_folder(self):
        if self.current:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.jobs.path(self.current))))
    def display_error(self,message):
        self.load_failed=True;self.error.setText('3D display: '+message);self.pages.setCurrentIndex(0)

    def show_previews(self,rows):
        stamp=json.dumps(rows,sort_keys=True)
        if stamp==self.preview_stamp:return
        self.preview_stamp=stamp
        self.selected=next((r['validated'] for r in rows if r.get('selected') and r.get('validated')),None)
        self.viewer.selected=self.selected
        if self.loaded:return
        while self.preview_layout.count():
            widget=self.preview_layout.takeAt(0).widget()
            if widget:widget.deleteLater()
        for i,row in enumerate(rows):self.preview_layout.addWidget(StructureCard(row),i//3,i%3)
        if rows:self.pages.setCurrentIndex(1)

    def refresh(self):
        if self.closed:return
        self.start.setEnabled(not self.jobs.active and not self.stopping)
        self.stop.setEnabled(bool(self.jobs.active) and not self.stopping)
        self.properties.setEnabled(not self.jobs.active)
        if self.stopping and self.stopping.done():
            try:self.stopping.result()
            except ValueError:pass
            self.stopping=None
        if not self.current:return
        folder=self.jobs.path(self.current);state=self.jobs.state(self.current)
        self.run_state.setText('  '+state['state'].title())
        suffix=f' · {self.viewer.phase} · {self.viewer.sim_time:.2f} s' if self.loaded else ''
        self.statusBar().showMessage(state['message']+suffix)
        if self.preview_future and self.preview_future[1].done():
            key,future=self.preview_future;self.preview_future=None
            try:
                rows=future.result()
                if key==self.current:self.show_previews(rows)
            except Exception as error:self.error.setText('Structure display: '+str(error))
        if not self.preview_future and (not self.loaded or not self.selected):
            self.preview_future=(self.current,self.pool.submit(read_previews,folder/'planning'))
        if self.model_future and self.model_future[1].done():
            key,future=self.model_future;self.model_future=None
            try:
                model=future.result()
                if key==self.current:
                    self.pages.setCurrentIndex(2);self.viewer.attach(folder,model)
                    self.viewer.selected=self.selected;self.loaded=True
            except Exception as error:self.display_error(str(error))
        if not self.loaded and not self.load_failed and not self.model_future and (folder/'live.json').exists():
            self.model_future=(self.current,self.pool.submit(load_display_model,folder))
        result=state.get('result')
        if result:
            stamp=json.dumps(result,sort_keys=True)
            if stamp!=self.result_stamp:
                self.result_stamp=stamp;self.outcome.setText(('Succeeded' if result['success'] else 'Failed')+' · '+str(result.get('reason','')).replace('_',' '))
                preparation=result.get('preparation_s');control=result.get('control',{});ms=control.get('max_ms',control.get('maximum_ms'))
                plan='—' if preparation is None else f'{preparation:.2f} s';timing='—' if ms is None else f'{ms:.2f} ms'
                self.metrics.setText(f"Modules {result.get('modules','—')}     Planning / setup {plan}     Max. command time {timing}")
                self.detail.setPlainText(json.dumps(result,indent=2))
        elif state['state'] in ('cancelled','error'):self.outcome.setText(state['message'])
        self.follow.setChecked(self.viewer.follow)

    def closeEvent(self,event):
        if self.jobs.active:
            answer=QMessageBox.question(self,'Stop execution?','Closing this window stops the current run. Close anyway?')
            if answer!=QMessageBox.StandardButton.Yes:event.ignore();return
            try:self.jobs.cancel(self.jobs.active)
            except ValueError:pass
        self.closed=True;self.poll.stop();self.draw_timer.stop();self.viewer.release()
        self.pool.shutdown(wait=True,cancel_futures=True)
        if self.jobs.monitor_thread:self.jobs.monitor_thread.join(timeout=6)
        event.accept()


def run():
    import sys
    fmt=QSurfaceFormat();fmt.setDepthBufferSize(24);fmt.setStencilBufferSize(8)
    fmt.setVersion(2,1);fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app=QApplication(sys.argv);app.setApplicationName('A-MSRR');app.setStyle('Fusion');app.setStyleSheet(STYLE)
    window=TaskWindow();window.show()
    return app.exec()
