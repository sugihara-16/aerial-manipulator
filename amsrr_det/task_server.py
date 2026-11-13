"""Local-only task console with one bounded subprocess job at a time."""
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .task_spec import CATALOG,COMMON,make_spec,validate_spec,form_values
from .task_execution import write
from .planning_preview import read_previews

ROOT=Path(__file__).resolve().parents[1]


class Jobs:
    def __init__(self,folder,*,render_frames=True):
        self.render_frames=render_frames
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock();self.active=None;self.process=None;self.monitor_thread=None

    def start(self,spec):
        spec=validate_spec(spec)
        with self.lock:
            if self.active:raise ValueError('A run is active. Wait for it to finish or stop it.')
            key=uuid.uuid4().hex;out=self.folder/key;out.mkdir();write(out/'task.json',spec)
            write(out/'status.json',dict(state='queued',message='Preparing the run'))
            log=open(out/'execution.log','w');env=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            (out/'planning').mkdir()
            env['AMSRR_PLANNING_PREVIEW']=str(out/'planning')
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','scripts.run_task',str(out/'task.json'),str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.active=key;self.process=proc
            self.monitor_thread=threading.Thread(target=self.monitor,args=(key,proc,log,env),daemon=True)
            self.monitor_thread.start()
            return key

    def monitor(self,key,proc,log,env):
        out=self.folder/key;renderer=None;render_log=None;began=time.monotonic();timeout=False
        try:
            while proc.poll() is None:
                if self.render_frames and renderer is None and (out/'live.json').exists():
                    render_log=open(out/'render.log','w')
                    renderer=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','scripts.render_task_live',str(out)],cwd=ROOT,env=env,stdout=render_log,stderr=subprocess.STDOUT,start_new_session=True)
                if time.monotonic()-began>600 or (not (out/'live.json').exists() and time.monotonic()-began>45):
                    timeout=True;self.kill(proc);break
                time.sleep(.3)
            proc.wait()
            if timeout:write(out/'status.json',dict(state='error',message='Stopped after exceeding the computation time limit'))
            elif self.state(key).get('state') not in ('finished','error','cancelled'):
                write(out/'status.json',dict(state='error',message='The execution process exited. Check the log.'))
            # Let the display consume the final published state, independently.
            if renderer:time.sleep(.5)
        finally:
            if renderer:self.kill(renderer);renderer.wait()
            if render_log:render_log.close()
            log.close()
            with self.lock:
                if self.active==key:self.active=None;self.process=None

    @staticmethod
    def kill(proc):
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=2.)

    def cancel(self,key):
        with self.lock:
            if key!=self.active:raise ValueError('This run has already ended')
            write(self.folder/key/'status.json',dict(state='cancelled',message='Cancelled by user'))
            self.kill(self.process)

    def path(self,key):
        if not re.fullmatch('[0-9a-f]{32}',key):raise ValueError('Invalid run ID')
        p=self.folder/key
        if not p.is_dir():raise ValueError('Run not found')
        return p

    def state(self,key):
        p=self.path(key)
        try:s=json.loads((p/'status.json').read_text())
        except (OSError,ValueError):s=dict(state='queued',message='Preparing')
        # Display earlier runs in English without rewriting their saved records.
        messages={'queued':'Preparing the run','planning':'Generating structure',
                  'assembling':'Executing takeoff, assembly, and stable hovering',
                  'running':'Simulation running','cancelled':'Cancelled by user',
                  'finished':'Succeeded' if (s.get('result') or {}).get('success') else 'Failed'}
        s['message']=messages.get(s['state'],s.get('message',''))
        if (p/'live_status.json').exists():
            try:s['live']=json.loads((p/'live_status.json').read_text())
            except (OSError,ValueError):pass
        if s['state']=='planning' and (p/'live.json').exists():s.update(state='running',message='Simulation running')
        try:
            spec=json.loads((p/'task.json').read_text());s['task_label']=CATALOG[spec['task']]['label']
            s['mode_label']={'task':'Assembled start','assembly':'Assembly only','integrated':'Assembly to task'}[spec['execution']['mode']]
        except (OSError,ValueError,KeyError):pass
        s.update(id=key,has_frame=(p/'frame.jpg').exists(),active=key==self.active)
        if (p/'render.log').exists() and (p/'render.log').stat().st_size and not s['has_frame']:s['display_warning']='Display initialization failed. Execution is continuing.'
        return s


def serve(port=8765,folder=None):
    jobs=Jobs(folder or ROOT/'artifacts/task_console')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,value,status=200,kind='application/json'):
            raw=json.dumps(value,ensure_ascii=False).encode() if kind=='application/json' else value
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            try:
                path=urlparse(self.path).path
                if path=='/':return self.reply((ROOT/'web/task_console/index.html').read_bytes(),kind='text/html; charset=utf-8')
                if path=='/api/catalog':return self.reply(dict(tasks=CATALOG,common=COMMON))
                if path=='/api/jobs':
                    paths=sorted(jobs.folder.glob('*/task.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]
                    return self.reply(dict(active=jobs.active,jobs=[jobs.state(p.parent.name) for p in paths]))
                match=re.fullmatch('/api/jobs/([0-9a-f]{32})(?:/([a-z_.]+))?',path)
                if not match:return self.reply(dict(error='Not found'),404)
                key,name=match.groups()
                if not name:return self.reply(jobs.state(key))
                if name=='planning.json':return self.reply(read_previews(jobs.path(key)/'planning'))
                if name=='stream.mjpg':
                    p=jobs.path(key)/'frame.jpg'
                    self.send_response(200)
                    self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control','no-store')
                    self.end_headers()
                    previous=None
                    try:
                        while True:
                            if p.exists():
                                stamp=p.stat().st_mtime_ns
                                if stamp!=previous:
                                    raw=p.read_bytes();previous=stamp
                                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw+b'\r\n')
                                    self.wfile.flush()
                            if key!=jobs.active:break
                            time.sleep(1/60)
                        self.wfile.write(b'--frame--\r\n')
                    except (BrokenPipeError,ConnectionResetError):pass
                    return
                allowed={'task.json':'application/json','requirements.json':'application/json','result.json':'application/json','frame.jpg':'image/jpeg','execution.log':'text/plain; charset=utf-8'}
                if name not in allowed:return self.reply(dict(error='Unsupported file'),404)
                p=jobs.path(key)/name
                if not p.exists():return self.reply(dict(error='Not generated yet'),404)
                kind=allowed[name];raw=p.read_bytes()
                return self.reply(json.loads(raw) if kind=='application/json' else raw,kind=kind)
            except (OSError,ValueError) as e:self.reply(dict(error=str(e)),400)
        def do_POST(self):
            try:
                host=self.headers.get('Host','');origin=self.headers.get('Origin')
                if host not in (f'127.0.0.1:{port}',f'localhost:{port}') or (origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}')):return self.reply(dict(error='Use the local console to perform this action'),403)
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=65536:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(size))
                if self.path=='/api/preview':return self.reply(make_spec(body['task'],body['values'],body['mode']))
                if self.path=='/api/validate':
                    s=validate_spec(body);return self.reply(dict(spec=s,values=form_values(s)))
                if self.path=='/api/jobs':return self.reply(dict(id=jobs.start(body)),201)
                match=re.fullmatch('/api/jobs/([0-9a-f]{32})/cancel',self.path)
                if match:jobs.cancel(match[1]);return self.reply(dict(ok=True))
                return self.reply(dict(error='Not found'),404)
            except (ValueError,KeyError,TypeError) as e:self.reply(dict(error=str(e)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Task console: http://127.0.0.1:{port}',flush=True)
    def stop_server(*args):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,stop_server)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        key=jobs.active
        if key:
            try:jobs.cancel(key)
            except ValueError:pass
        if jobs.monitor_thread:jobs.monitor_thread.join(timeout=6.)
        server.server_close()
        signal.signal(signal.SIGTERM,previous)
