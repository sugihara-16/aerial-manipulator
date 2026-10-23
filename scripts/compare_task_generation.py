"""Paired generator comparisons; all modes use the unchanged physical screen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from build.frozen_structure_trial import trial
from amsrr_det.structure_workers import planning_context


def run(job):
    request,descriptor,modules,index,seed,mode,starts=job
    began=time.perf_counter()
    result=trial(request,modules,index,seed,starts=starts,task_wrench=True,
                 generator='legacy' if mode=='legacy' else 'task',
                 descriptor=descriptor if mode=='swapped' else None,
                 common_initialization=mode!='integrated')
    result.pop('plan');result.pop('prepared')
    return dict(case=request['id'],mode=mode,job_s=time.perf_counter()-began,**result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=8);parser.add_argument('--starts',type=int,default=4)
    parser.add_argument('--modules',type=int,nargs='+',default=[2,3,4,5]);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--modes',nargs='+',default=['legacy','correct','swapped','integrated'])
    parser.add_argument('--seed',type=int,default=528146);args=parser.parse_args()
    args.output.mkdir(exist_ok=False);panel=json.loads(args.panel.read_text());cases=panel['cases']
    (args.output/'panel.json').write_text(json.dumps(panel,indent=2))
    source_files=[*Path('amsrr_det').glob('*.py'),Path('build/frozen_structure_trial.py'),Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (args.output/'protocol.json').write_text(json.dumps(dict(seed=args.seed,trials=args.trials,starts=args.starts,
        modules=args.modules,modes=args.modes,workers=args.workers,source_hashes=hashes),indent=2))
    jobs=[(case,cases[(i+1)%len(cases)],n,t,args.seed+i,mode,args.starts)
          for i,case in enumerate(cases) for n in args.modules for t in range(args.trials) for mode in args.modes]
    began=time.perf_counter();summaries=[]
    with (args.output/'rows.jsonl').open('w') as stream,ProcessPoolExecutor(max_workers=args.workers,mp_context=planning_context()) as pool:
        for result in pool.map(run,jobs):
            stream.write(json.dumps(result)+'\n');stream.flush()
            summaries.append(dict(case=result['case'],mode=result['mode'],modules=result['modules'],trial=result['trial'],
                accepted=sum(r['accepted'] for r in result['rows']),starts=len(result['rows']),burden=result['burden'],wall_s=result['wall_s']))
            if len(summaries)%24==0:print(json.dumps(dict(completed=len(summaries),total=len(jobs),wall_s=time.perf_counter()-began)),flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-began,rows=summaries),indent=2))
    for mode in args.modes:
        rows=[r for r in summaries if r['mode']==mode]
        print(json.dumps(dict(mode=mode,accepted=sum(r['accepted'] for r in rows),starts=sum(r['starts'] for r in rows),
            worker_elapsed_s=sum(r['wall_s'] for r in rows),best={c['id']:min(r['burden'] for r in rows if r['case']==c['id']) for c in cases})),flush=True)


if __name__=='__main__':
    main()
