"""Describe the frozen extension panel; no held-out or real-time claim."""
import json,sys,collections,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.report_grasp_panel import report
root=Path(sys.argv[1]);panel=json.loads((root/'panel.json').read_text())
result=report(panel,[root],panel['configuration'])
rows=[]
for item in result['rows']:
 if item['result_path'] is None:continue
 p=Path(item['result_path']);d=json.loads(p.read_text());selection=p.parent/'selection.json'
 if not selection.exists():selection=p.parent/'selection_summary.json'
 s=json.loads(selection.read_text()) if selection.exists() else {}
 plan=p.parent/'generated_plan/plan.json';a=json.loads(plan.read_text()) if plan.exists() else {}
 rows.append(dict(case=item['case'],success=item['success'],reason=item['reason'],modules=d.get('modules',a.get('model',{}).get('modules')),
  planning_s=d.get('initial_planning_s'),wall_s=d.get('wall_s'),simulation_s=d.get('simulation_s'),
  evidence_complete=bool(s.get('selection_evidence') and s['selection_evidence']['complete']),
  selected_burden=s.get('selected_burden'),surfaces=a.get('surfaces'),edges=a.get('model',{}).get('edges'),
  phase_control_time_s=d.get('phase_control_time_s')))
def graph_key(row, labelled=False):
 n=row['modules'];edges=row['edges']
 if not n or edges is None:return None
 graph=[[] for _ in range(n)];surfaces=dict()
 for i,port in row['surfaces'] or []:surfaces.setdefault(i,[]).append(port)
 for e in edges:
  graph[e['a']].append((e['b'],e['port_a'],e['port_b']))
  graph[e['b']].append((e['a'],e['port_b'],e['port_a']))
 def branch(node,parent):
  children=sorted((pa+':'+pb+branch(child,node) if labelled else branch(child,node)) for child,pa,pb in graph[node] if child!=parent)
  return '('+(','.join(sorted(surfaces.get(node,[]))) if labelled else '')+'|'+','.join(children)+')'
 return min(branch(i,-1) for i in range(n))
def stats(values):
 values=[v for v in values if v is not None]
 return dict(min=min(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=max(values)) if values else None
output=dict(overall=result['overall'],complete_results=len(rows),input_role=panel['role'],
 untouched_holdout=False,module_counts=dict(collections.Counter(r['modules'] for r in rows)),
 success_module_counts=dict(collections.Counter(r['modules'] for r in rows if r['success'])),
 planning_seconds=stats([r['planning_s'] for r in rows]),
 execution_wall_over_simulation=stats([r['wall_s']/r['simulation_s'] for r in rows if r['wall_s'] and r['simulation_s']]),
 skeleton_types=len({graph_key(r) for r in rows if r['edges'] is not None}),
 port_and_pad_types=len({graph_key(r,True) for r in rows if r['edges'] is not None}),
 complete_selection_evidence=sum(r['evidence_complete'] for r in rows),
 all_results_same_runtime_sources=result['all_results_same_runtime_sources'],
 all_results_same_execution_settings=result['all_results_same_execution_settings'],
 failure_concentration=result['failure_concentration'],strata=result['strata'],rows=rows)
(root/'analysis.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:output[k] for k in ['overall','module_counts','planning_seconds','complete_selection_evidence']}))
