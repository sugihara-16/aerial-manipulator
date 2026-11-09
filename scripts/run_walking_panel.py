"""Frozen-source independent walking evaluations, with measured control delay."""
import argparse,json,os,signal,shutil,subprocess,time
import numpy as np
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes,environment


def run(panel_path,out,indices=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    panel=json.loads(Path(panel_path).read_text());cases=panel['cases']
    if indices is not None:cases=[cases[i] for i in indices]
    source=source_hashes()
    for name in source:
        saved=out/'source_snapshot'/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,saved)
    (out/'sources.json').write_text(json.dumps(source,indent=2))
    (out/'environment.json').write_text(json.dumps(environment(),indent=2))
    (out/'panel.json').write_text(json.dumps(dict(panel,cases=cases),indent=2))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
    rows=[];started=time.perf_counter()
    for case in cases:
        if source_hashes()!=source:raise RuntimeError('source changed during panel')
        task=case['task']
        cmd=['.venv/bin/python','-m','scripts.run_walking_task','--modules',str(task['modules']),
             '--speed',str(task['speed']),'--curvature',str(task['curvature']),'--seed',str(case['seed']),
             '--out',str(out/case['id'])]
        with (out/(case['id']+'.log')).open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM);process.wait();code=-15
        path=out/case['id']/'result.json'
        row=json.loads(path.read_text()) if path.exists() else dict(task_success=False,reason='runner_error',exit_code=code)
        row.update(id=case['id'],task=task,seed=case['seed']);rows.append(row)
        summary=dict(completed=len(rows),total=len(cases),successes=sum(r['task_success'] for r in rows),
                     wall_s=time.perf_counter()-started,source_changes=source_hashes()!=source,results=rows)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(case['id'],row['task_success'],row.get('reason'),round(row.get('planning_s',0),3),
              [key for key,val in row.get('checks',{}).items() if not val],flush=True)
        if summary['source_changes']:raise RuntimeError('source changed during evaluation')
        # Every individual calculation still has its original deadline.
        # Stop for repairs once even success in every remaining case cannot
        # reach the requested fraction; isolated timeouts remain failures.
        required=int(np.ceil(panel['acceptance']['minimum_success_fraction']*len(cases)))
        if summary['successes']+len(cases)-len(rows)<required or row.get('reason')=='runner_error':break
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--panel',default='configs/evaluation/walking36.json');p.add_argument('--ids');a=p.parse_args()
    run(a.panel,a.out,None if a.ids is None else list(map(int,a.ids.split(','))))
