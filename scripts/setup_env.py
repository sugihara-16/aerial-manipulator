#!/usr/bin/env python3
"""Create only this repository's environment; works without distro ensurepip."""
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile
import argparse

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--wheelhouse',type=Path,help='Optional offline wheels matching this architecture/Python')
args=parser.parse_args()
environment=os.environ.copy();environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
target=root/'.venv'
if not (target/'bin/python').exists():venv.EnvBuilder(with_pip=False).create(target)
python=target/'bin/python'
check=subprocess.run([str(python),'-m','pip','--version'],env=environment,capture_output=True)
if check.returncode:
    bootstrap=root/'.bootstrap';bootstrap.mkdir(exist_ok=True)
    wheel_path=args.wheelhouse/'pip-26.2.1-py3-none-any.whl' if args.wheelhouse else bootstrap/'pip-26.2.1-py3-none-any.whl'
    if not wheel_path.exists():
        if args.wheelhouse:raise SystemExit('Offline wheelhouse needs pip-26.2.1-py3-none-any.whl')
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','pip==26.2.1','-d',str(bootstrap)],env=environment,check=True)
    site=subprocess.check_output([str(python),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],env=environment,text=True).strip()
    with zipfile.ZipFile(wheel_path) as wheel:wheel.extractall(site)
options=['--no-index','--find-links',str(args.wheelhouse.resolve())] if args.wheelhouse else []
subprocess.run([str(python),'-m','pip','install',*options,'-r',str(root/'requirements.lock.txt')],env=environment,check=True)
print('Environment ready:',target)
