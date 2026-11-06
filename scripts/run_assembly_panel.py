"""Frozen assembly evaluations with bounded case duration and fixed CPU lanes."""
import argparse,concurrent.futures,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes,environment


def run(out,ids=None,pairs=((0,2),(4,6),(8,10))):
    os.sched_setaffinity(0,{31})
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    panel=json.loads(Path('configs/evaluation/assembly_generated.json').read_text())
    cases=panel['cases'] if ids is None else [c for c in panel['cases'] if c['id'] in ids]
    if ids is not None and len(cases)!=len(ids):raise ValueError('unknown or duplicated case')
    hashes=source_hashes()
    for name in hashes:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    (out/'sources.json').write_text(json.dumps(hashes,indent=2))
    (out/'environment.json').write_text(json.dumps(environment(),indent=2))
    (out/'panel.json').write_text(json.dumps(dict(panel,cases=cases,cpu_pairs=pairs),indent=2))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
    def lane(index):
        rows=[];cpus=pairs[index]
        for case in cases[index::len(pairs)]:
            if source_hashes()!=hashes:raise RuntimeError('runtime changed during evaluation')
            if hashlib.sha256(Path(case['source']).read_bytes()).hexdigest()!=case['sha256']:
                raise RuntimeError('input structure changed')
            folder=out/case['id'];folder.parent.mkdir(parents=True,exist_ok=True)
            cmd=['.venv/bin/python','-m','scripts.run_assembly_task',case['id'],str(folder),
                 '--control-cpu',str(cpus[0]),'--worker-cpu',str(cpus[1])]
            with (folder.parent/(folder.name+'.log')).open('w') as log:
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM);process.wait();code=-15
            path=folder/'result.json'
            row=json.loads(path.read_text()) if path.exists() else dict(success=False,reason='runner_timeout' if code==-15 else 'runner_error',exit_code=code)
            row.update(case=case['id'],task=case['task'],modules=case['modules'],cpu_pair=cpus)
            rows.append(row);(out/f'lane{index}.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:row.get(k)for k in ['case','modules','success','reason','planning_s','assembly_s','publication_max_s']}),flush=True)
        return rows
    began=time.perf_counter();rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as executor:
        futures=[executor.submit(lane,i) for i in range(len(pairs))]
        for future in concurrent.futures.as_completed(futures):rows.extend(future.result())
    changed=source_hashes()!=hashes
    result=dict(completed=len(rows),total=len(cases),successes=sum(r['success']for r in rows),
                actual_assemblies=sum(r['modules']>1 for r in rows),assembly_successes=sum(r['modules']>1 and r['success']for r in rows),
                wall_s=time.perf_counter()-began,source_changes=changed,results=sorted(rows,key=lambda r:r['case']))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    if changed:raise RuntimeError('runtime changed during evaluation')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out');parser.add_argument('--ids');parser.add_argument('--pairs',default='0:2,4:6,8:10');args=parser.parse_args()
    run(args.out,None if args.ids is None else args.ids.split(','),[tuple(map(int,p.split(':')))for p in args.pairs.split(',')])
