#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
prefix=Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/cmeel.prefix'
if not prefix.is_dir() or Path(sys.prefix).resolve()!=root/'.venv':raise SystemExit('Run with this repository .venv/bin/python')
env=os.environ.copy()
for key in ['PYTHONPATH','PYTHONHOME','CMAKE_PREFIX_PATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH']:
    env.pop(key,None)
cmake=root/'.venv/bin/cmake'
subprocess.run([str(cmake),'-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_PREFIX_PATH={prefix}',
               f'-DEigen3_DIR={prefix}/share/eigen3/cmake',f'-Durdfdom_headers_DIR={prefix}/lib/urdfdom_headers/cmake'],env=env,check=True)
subprocess.run([str(cmake),'--build',str(root/'build'),'-j1'],env=env,check=True)
sources=[root/'CMakeLists.txt',*sorted((root/'native').glob('*')),*sorted((root/'third_party').rglob('*.hpp'))]
manifest={'binary_sha256':hashlib.sha256((root/'build/det_rollout').read_bytes()).hexdigest(),
          'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(root/'build/native_identity.json').write_text(json.dumps(manifest,indent=2)+'\n')
