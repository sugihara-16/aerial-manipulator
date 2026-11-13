"""Verify and restore the archived evidence without overwriting different files."""
import argparse,hashlib,json,shutil,tarfile,tempfile
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('groups',nargs='*',help='desktop, transitions, walking; default: all')
p.add_argument('--verify-only',action='store_true');args=p.parse_args()
known={a['group'] for a in manifest['archives']}
if set(args.groups)-known:p.error('Unknown archive group')
def digest(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for a in manifest['archives']:
 if args.groups and a['group'] not in args.groups:continue
 with tempfile.TemporaryFile() as combined:
  for part in a['parts']:
   path=repo/part['path']
   with path.open('rb') as f:
    if path.stat().st_size!=part['bytes'] or digest(f)!=part['sha256']:raise ValueError(f'Part checksum mismatch: {path}')
    f.seek(0);shutil.copyfileobj(f,combined)
  combined.seek(0)
  if digest(combined)!=a['archive_sha256']:raise ValueError('Archive checksum mismatch')
  for write_files in ([False] if args.verify_only else [False,True]):
   combined.seek(0);seen=set()
   with tarfile.open(fileobj=combined,mode='r:xz') as tf:
    for member in tf:
     if not member.isfile() or member.name not in a['files']:raise ValueError('Unexpected archive entry')
     target=(repo/member.name).resolve()
     if not target.is_relative_to(repo/'artifacts'):raise ValueError('Path outside artifacts')
     expected=a['files'][member.name];seen.add(member.name)
     with tf.extractfile(member) as f:
      if not write_files:
       if digest(f)!=expected:raise ValueError(f'File checksum mismatch: {member.name}')
       if target.exists() and not args.verify_only:
        with target.open('rb') as existing:
         if digest(existing)!=expected:raise ValueError(f'Refusing to overwrite changed file: {target}')
      elif not target.exists():
       target.parent.mkdir(parents=True,exist_ok=True)
       with target.open('wb') as out:shutil.copyfileobj(f,out)
       target.chmod(member.mode & 0o777)
   if seen!=set(a['files']):raise ValueError('Missing archive files')
 print(a['group'],len(a['files']),'files verified' if args.verify_only else 'files restored')
