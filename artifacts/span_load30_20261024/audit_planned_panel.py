"""Verify frozen inputs, runtime snapshots, seeds and planned structure diversity."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
from scripts.report_structure_selection import signature
root=Path(sys.argv[1]);panel=json.loads((root/'panel.json').read_text());seeds=json.loads((root/'trial_seeds.json').read_text())
rows=[];manifests=[];hash_errors=[];input_errors=[];seed_errors=[]
for i,case in enumerate(panel['cases']):
    name=case['id'];folder=root/name
    request=root/'requests'/f'{name}.json'
    if request.exists() and json.loads(request.read_text())!=case:input_errors.append(name)
    mf=folder/'source_snapshot/manifest.json'
    if mf.exists():
        manifest=json.loads(mf.read_text())
        manifests.append({k:v for k,v in manifest.items() if k.startswith(('amsrr_det/','build/','configs/','native/','assets/generated/'))})
        for path,digest in manifest.items():
            if hashlib.sha256((mf.parent/path).read_bytes()).hexdigest()!=digest:hash_errors.append([name,path])
    sp=folder/'selection.json'
    if not sp.exists():sp=folder/'selection_summary.json'
    if not sp.exists():continue
    sel=json.loads(sp.read_text())
    if sel.get('seed')!=seeds[name] or seeds[name]!=2026102404+i:seed_errors.append(name)
    plan_file=folder/'generated_plan/plan.json'
    if not plan_file.exists():continue
    plan=json.loads(plan_file.read_text());n=plan['model']['modules'];edges=plan['model']['edges'];surfaces=plan['surfaces']
    degree=Counter(node for edge in edges for node in [edge['a'],edge['b']])
    ready=folder/'execution_ready.json'
    rows.append(dict(case=name,modules=n,contacts=len(surfaces),
        skeleton=repr(signature(n,edges,ports=False)),ports=repr(signature(n,edges,surfaces)),
        branched=max(degree.values(),default=0)>2,pitch=any(p.startswith('pitch')for _,p in surfaces),
        burden=sel.get('selected_burden'),evidence_complete=bool((sel.get('selection_evidence')or{}).get('complete')),
        planning_s=json.loads(ready.read_text())['initial_planning_s'] if ready.exists() else None))
result=dict(rows=rows,plans=len(rows),counts=dict(Counter(r['modules']for r in rows)),
    skeletons=len({r['skeleton']for r in rows}),port_structures=len({r['ports']for r in rows}),
    branched=sum(r['branched']for r in rows),pitch_grasps=sum(r['pitch']for r in rows),
    snapshots=len(manifests),same_runtime=all(m==manifests[0]for m in manifests),
    hash_errors=hash_errors,input_errors=input_errors,seed_errors=seed_errors,
    planning_range_s=[min(r['planning_s']for r in rows if r['planning_s']is not None),
                      max(r['planning_s']for r in rows if r['planning_s']is not None)] if rows else [])
(root/'planned_diversity.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
