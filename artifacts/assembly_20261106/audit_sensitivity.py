"""Create audit metadata over the unchanged finer-step outputs."""
import json,sys
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes
main=Path('artifacts/assembly_20261106/full219_central27');root=main.parent/'halfstep6_central27';out=main.parent/'halfstep6_audit_central27';out.mkdir(exist_ok=True)
rows=json.load(open(root/'summary.json'));source=json.load(open(main/'panel.json'));cases={r['id']:r for r in source['cases']}
for r in rows:
 r['task']=cases[r['case']]['task']
 target=out/r['case'];target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():target.symlink_to((root/r['case']).resolve(),target_is_directory=True)
for name in ['source_snapshot','sources.json']:
 target=out/name
 if not target.exists():target.symlink_to((main/name).resolve(),target_is_directory=(name=='source_snapshot'))
(out/'panel.json').write_text(json.dumps(dict(source,cases=[cases[r['case']]for r in rows]),indent=2))
(out/'summary.json').write_text(json.dumps(dict(results=rows,source_changes=source_hashes()!=json.load(open(main/'sources.json'))),indent=2))
print(out)
