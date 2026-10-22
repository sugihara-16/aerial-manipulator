#!/usr/bin/env python3
"""Package only this repository and official offline dependencies; no deployment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wheelhouse',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    names=['AGENTS.md','README.md','CMakeLists.txt','requirements.txt','requirements.lock.txt',
           'amsrr_det','native','scripts','tests','assets','configs','third_party','references','for_codex']
    sources=[]
    for name in names:
        source=ROOT/name
        for path in (sorted(source.rglob('*')) if source.is_dir() else [source]):
            if path.is_symlink():raise ValueError('source package must not follow symlinks')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':sources.append(path)
    archive=args.output/'amsrr_det_source.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for path in sources:tar.add(path,arcname='amsrr_det/'+str(path.relative_to(ROOT)),recursive=False)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),hardware_tested=False,
        purpose='Portable isolated build and numerical/CPU diagnostic; no motor or hardware I/O code',
        source_archive=dict(path=archive.name,sha256=sha(archive),size=archive.stat().st_size),
        sources={str(path.relative_to(ROOT)):sha(path) for path in sources})
    if args.wheelhouse:
        wheels=sorted(args.wheelhouse.glob('*.whl'))
        if not wheels or any('x86_64' in p.name for p in wheels):raise ValueError('expected aarch64/architecture-independent wheels')
        target=args.output/'aarch64_python310_wheels.tar'
        with tarfile.open(target,'w') as tar:
            for path in wheels:tar.add(path,arcname='wheels/'+path.name,recursive=False)
        manifest['wheel_archive']=dict(path=target.name,sha256=sha(target),size=target.stat().st_size)
        manifest['wheels']={path.name:sha(path) for path in wheels}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['sources','wheels']},indent=2))


if __name__=='__main__':main()
