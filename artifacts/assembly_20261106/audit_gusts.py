import json
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes
main=Path('artifacts/assembly_20261106/full219_central27');out=main.parent/'gust_audit_central27';out.mkdir(exist_ok=True)
source=json.load(open(main/'panel.json'));case=next(c for c in source['cases']if c['id']=='grasp/box_y0_0');rows=[];cases=[]
for force in [2,3]:
 label=f'gust_{force}n';p=main.parent/f'gust_{force}n_central27';r=json.load(open(p/'result.json'));r.update(case=label,task='grasp');rows.append(r);cases.append(dict(case,id=label))
 target=out/label
 if not target.exists():target.symlink_to(p.resolve(),target_is_directory=True)
for name in ['source_snapshot','sources.json']:
 target=out/name
 if not target.exists():target.symlink_to((main/name).resolve(),target_is_directory=name=='source_snapshot')
(out/'panel.json').write_text(json.dumps(dict(source,cases=cases),indent=2))
(out/'summary.json').write_text(json.dumps(dict(results=rows,source_changes=source_hashes()!=json.load(open(main/'sources.json'))),indent=2))
