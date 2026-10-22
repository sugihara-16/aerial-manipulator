"""Compile only the local batched geometry kernel, outside runtime planning."""
import hashlib,json,os,subprocess,sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
prefix=Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/cmeel.prefix'
if Path(sys.prefix).resolve()!=root/'.venv':raise SystemExit('use this repository .venv')
env=os.environ.copy()
for key in ['PYTHONPATH','PYTHONHOME','CMAKE_PREFIX_PATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH']:env.pop(key,None)
cmake=str(root/'.venv/bin/cmake');destination=root/'build/geometry_kernel'
subprocess.run([cmake,'-S',str(root),'-B',str(destination),'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_PREFIX_PATH={prefix}',
                f'-DEigen3_DIR={prefix}/share/eigen3/cmake',f'-Durdfdom_headers_DIR={prefix}/lib/urdfdom_headers/cmake'],env=env,check=True)
subprocess.run([cmake,'--build',str(destination),'--target','geometry_kernel','-j1'],env=env,check=True)
sources=['native/geometry_kernel.cpp','CMakeLists.txt']
manifest=dict(binary_sha256=hashlib.sha256((destination/'libgeometry_kernel.so').read_bytes()).hexdigest(),
              sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})
(destination/'identity.json').write_text(json.dumps(manifest,indent=2))
