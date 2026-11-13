"""Reuse the assembly state auditor on the integrated panel's assembly stage."""
import json
from pathlib import Path
from scripts.report_assembly_panel import report
root=Path('artifacts/walking_integrated20_20261112/full36').resolve()
view=root/'assembly_audit';view.mkdir(exist_ok=True)
cases=[];rows=[]
for item in json.loads((root/'panel.json').read_text())['cases']:
 folder=root/item['id']/'assembly'
 if not (folder/'result.json').exists():continue
 case=json.loads((folder/'input.json').read_text());case.update(id=item['id'],task='walking')
 row=json.loads((folder/'result.json').read_text());row.update(case=item['id'],task='walking')
 cases.append(case);rows.append(row)
 link=view/item['id']
 if not link.exists():link.symlink_to(folder,target_is_directory=True)
for name in ['sources.json','source_snapshot']:
 p=view/name
 if not p.exists():p.symlink_to(root/name,target_is_directory=name=='source_snapshot')
(view/'panel.json').write_text(json.dumps(dict(cases=cases),indent=2))
(view/'summary.json').write_text(json.dumps(dict(results=rows,source_changes=json.loads((root/'summary.json').read_text())['source_changes']),indent=2))
r=report(view)
print(json.dumps({k:v for k,v in r.items() if k not in ['checks','failure_concentration']},indent=2))
