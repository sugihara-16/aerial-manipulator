"""Package adopted evidence, preserving bytes and verifying every archived file."""
import hashlib,json,lzma,tarfile
from pathlib import Path
root=Path('artifacts/release_20261114')
groups={
 'desktop':[Path('artifacts/desktop_gui_20261111')],
 'transitions':[Path('artifacts/gui_direct_transition_20261111'),Path('artifacts/gui_recordings_v2_20261111')],
 'walking':[Path('artifacts/walking_integrated_20261112'),Path('artifacts/walking_integrated20_20261112'),Path('artifacts/walking_speed_recheck_20261113')],
}
video_dirs={'grasp_clear','valve_clear','walking_clear','walking_speed04'}
def included(p,base):
 if p.is_symlink() or not p.is_file():return False
 if any(x in p.parts for x in ['__pycache__','.pytest_cache']):return False
 if p.name in ['live.bin','live.json','live.xml','active.json']:return False
 if p.suffix=='.mp4':return base.name=='gui_direct_transition_20261111' and p.parent.name in video_dirs
 return True
manifest=dict(scope='Adopted desktop and transition implementation evidence; both complete walking panels (including failures); fixed preparation recheck. Final four direct-transition videos retained byte-for-byte. Earlier recording videos and volatile live IPC files omitted; originals remain locally.',archives=[])
for group,bases in groups.items():
 files=sorted({p for base in bases for p in base.rglob('*') if included(p,base)})
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 archive=root/(group+'.tar.xz')
 print('PACK',group,len(files),sum(p.stat().st_size for p in files),flush=True)
 with tarfile.open(archive,'w:xz',preset=1) as tf:
  for p in files:tf.add(p,arcname=str(p),recursive=False)
 verified={}
 with tarfile.open(archive,'r:xz') as tf:
  for item in tf:
   f=tf.extractfile(item)
   if f is not None:verified[item.name]=hashlib.sha256(f.read()).hexdigest()
 assert verified==hashes,group
 parts=[]
 if archive.stat().st_size>48*1024*1024:
  with archive.open('rb') as f:
   i=0
   while chunk:=f.read(48*1024*1024):
    i+=1;p=Path(str(archive)+f'.part{i:02d}');p.write_bytes(chunk);parts.append(p)
 else:parts=[archive]
 row=dict(group=group,files=hashes,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,parts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in parts],verified_files=len(verified))
 manifest['archives'].append(row)
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('VERIFIED',group,len(verified),row['archive_bytes'],flush=True)
(root/'tracked_archives.txt').write_text('\n'.join(p['path'] for a in manifest['archives'] for p in a['parts'])+'\n')
