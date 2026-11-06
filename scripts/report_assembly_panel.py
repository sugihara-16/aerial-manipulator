"""Recheck capture/hover/final states and aggregate the frozen assembly panel."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
from scipy.stats import fisher_exact
from amsrr_det.assembly import make_assembly_plant,dock_error,ready_to_attach
from amsrr_det.assembly_plan import initial_xy


def report(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());panel=json.loads((root/'panel.json').read_text())
    rows={r['case']:r for r in summary['results']};issues=[];delays=[];pub=[];intervals=[];checks=[]
    if summary['source_changes']:issues.append('runtime changed')
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        if hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()!=digest:issues.append('snapshot '+name)
    for case in panel['cases']:
        folder=root/case['id'];row=rows[case['id']]
        if not (folder/'final.npz').exists():continue
        plant=make_assembly_plant(case['modules'],case['edges']);plant.mj=mujoco.MjModel.from_xml_path(str(folder/'physical.xml'))
        m=plant.mj;d=mujoco.MjData(m);final=np.load(folder/'final.npz');planned=np.load(folder/'planned.npz')
        qi=np.array([int(m.jnt_qposadr[m.joint(plant.pin.names[b['joint']]).id])for b in plant.meta['bases']])
        vi=np.array([int(m.jnt_dofadr[m.joint(plant.pin.names[b['joint']]).id])for b in plant.meta['bases']])
        xy,yaw=initial_xy(case['modules'],case['initial_seed']);plan=json.loads((folder/'plan.json').read_text());assignment=plan['physical_module_assignment']
        expected_quat=np.c_[np.cos(yaw[assignment]/2),np.zeros((len(qi),2)),np.sin(yaw[assignment]/2)]
        ok_initial=(sorted(assignment)==list(range(case['modules'])) and
                    np.allclose(planned['initial'][qi[:,None]+np.arange(2)],xy[assignment]) and
                    np.allclose(abs(np.sum(planned['initial'][qi[:,None]+np.arange(3,7)]*expected_quat,axis=1)),1.))
        events=json.loads((folder/'events.json').read_text());captures=[e for e in events if e.get('event')=='latch'];capture_ok=True
        for e in captures:
            d.qpos[:]=e['qpos'];d.qvel[:]=e['qvel'];q,v=plant.observation(d)
            actual=dock_error(plant,q,v,case['edges'][e['edge']])
            capture_ok &= ready_to_attach(actual) and all(abs(actual[k]-e['errors'][k])<1e-8 for k in e['errors'])
            if 'central_mask' in row:
                samples=e.get('samples',[])
                capture_ok &= bool(samples and samples[-1]['t']-samples[0]['t']>=.1-1e-8 and
                                   all(ready_to_attach(s)for s in samples) and np.all(np.diff([s['t']for s in samples])>0))
        hover=next((e for e in events if e.get('stage')=='TRANSIT'),None);hover_ok=False
        if hover:
            q=np.array(hover['qpos']);v=np.array(hover['qvel']);goal=planned['initial'].copy();goal[qi+2]=1.5
            hover_ok=(max(np.linalg.norm(q[k:k+3]-goal[k:k+3])for k in qi)<.04 and
                      max(np.linalg.norm(v[j:j+3])for j in vi)<.06 and
                      max(np.linalg.norm(v[j+3:j+6])for j in vi)<.1 and
                      max(2*np.arccos(np.clip(abs(q[k+3:k+7]@goal[k+3:k+7]),0.,1.))for k in qi)<np.deg2rad(3.))
        d.qpos[:]=final['q'];d.qvel[:]=final['v'];d.eq_active[:]=final['seams'];q,v=plant.observation(d)
        errors=[dock_error(plant,q,v,e)for e in case['edges']]
        held=all(abs(e['axial_m'])<.003 and e['transverse_m']<.002 and e['attitude_rad']<np.deg2rad(.5)for e in errors)
        held &= max(np.linalg.norm(d.qvel[j:j+3])for j in vi)<.1 and max(np.linalg.norm(d.qvel[j+3:j+6])for j in vi)<.1
        elapsed=None if hover is None else float(final['time'])-hover['t']
        independent=bool(ok_initial and capture_ok and hover_ok and held and all(final['seams']) and
                         len(captures)==len(case['edges']) and len(set(e['edge']for e in captures))==len(captures) and elapsed<=30.+1e-8)
        central_ok=True
        if 'central_mask' in row:
            central_ok=(row['central_mask']==(1<<len(case['edges']))-1 and row['handoff_complete'] and
                        row['leaders']=={str(i):0 for i in range(case['modules'])})
            handoffs=json.loads((folder/'handoffs.json').read_text())
            central_ok &= all(e.get('first_output_difference',float('inf'))<1e-10 for e in handoffs)
            observations=folder/'hover_observations.npz'
            if observations.exists():
                obs=np.load(observations);selected=np.flatnonzero(obs['t']>=obs['t'][-1]-1.-1e-8)
                stable=bool(len(selected)>1 and obs['t'][selected[-1]]-obs['t'][selected[0]]>=1.-1e-8)
                goal=obs['reference'][selected[-1]]
                for i in selected:
                    q=obs['q'][i];v=obs['v'][i]
                    stable &= bool(obs['ready'][i] and np.allclose(obs['reference'][i],goal,rtol=0.,atol=1e-12) and
                        max(np.linalg.norm(q[k:k+3]-goal[k:k+3])for k in qi)<.04 and
                        max(2*np.arccos(np.clip(abs(q[k+3:k+7]@goal[k+3:k+7]),0.,1.))for k in qi)<np.deg2rad(3.) and
                        max(np.linalg.norm(v[j:j+3])for j in vi)<.06 and
                        max(np.linalg.norm(v[j+3:j+6])for j in vi)<.1)
                    d.qpos[:]=q;d.qvel[:]=v;pq,pv=plant.observation(d)
                    seam_errors=[dock_error(plant,pq,pv,e)for e in case['edges']]
                    stable &= all(abs(e['axial_m'])<.003 and e['transverse_m']<.002 and e['attitude_rad']<np.deg2rad(.5)for e in seam_errors)
                if handoffs:
                    stable &= all(e.get('central_start_s',float('inf'))+.6<=obs['t'][selected[0]]+1e-8 for e in handoffs)
                central_ok &= stable
            else:central_ok=False
            commands=folder/'handoff_commands.npz'
            if commands.exists():
                cmd=np.load(commands);nj=len(plant.meta['actuators'])
                limits=np.r_[[a['upper']for a in plant.meta['actuators']],[r['maximum']for r in plant.meta['rotors']]]
                lower=np.r_[[a['lower']for a in plant.meta['actuators']],np.zeros(len(plant.meta['rotors']))]
                if np.any(cmd['after']>limits+1e-9) or np.any(cmd['after']<lower-1e-9):issues.append(case['id']+' command outside input bounds')
                stale=cmd['mask']!=cmd['authority']
                if np.any(abs(cmd['after'][stale]-cmd['before'][stale])>1e-12):issues.append(case['id']+' stale topology changed output')
            elif case['edges'] and row['success']:issues.append(case['id']+' missing handover commands')
            independent &= bool(central_ok)
        timing=json.loads((folder/'timing.json').read_text());pub.extend(c['publication_s']for c in timing['calls']);intervals.extend(np.diff([c['t']for c in timing['calls']]))
        for job in timing['allocation']:
            if 'application_s' not in job:continue
            if job['application_s']+1e-8<job['available_s']:issues.append(case['id']+' applied before calculation finished')
            delays.append(job['application_s']-job['observation_s'])
        if row['success'] and not independent:issues.append(case['id']+' successful flag disagrees with state audit')
        if not capture_ok or not ok_initial:issues.append(case['id']+' invalid capture or initial state')
        checks.append(dict(case=case['id'],capture_ok=bool(capture_ok),initial_ok=bool(ok_initial),hover_ok=bool(hover_ok),central_ok=bool(central_ok),independent_success=independent))
    values=list(rows.values());success=sum(r['success']for r in values);groups={};contrasts=[]
    for key in ['task','modules']:
        for value in sorted(set(r[key]for r in values)):
            group=[r for r in values if r[key]==value];count=len(group);passed=sum(r['success']for r in group)
            groups[f'{key}:{value}']=dict(success=passed,total=count)
            if count<len(values):
                pv=float(fisher_exact([[count-passed,passed],[len(values)-count-(success-passed),success-passed]],alternative='greater').pvalue)
                contrasts.append(dict(group=f'{key}:{value}',p=pv))
    ordered=sorted(contrasts,key=lambda x:x['p']);previous=0.
    for i,c in enumerate(ordered):previous=max(previous,min(1.,(len(ordered)-i)*c['p']));c['holm_p']=previous
    result=dict(total=len(values),successes=success,assembly_total=sum(r['modules']>1 for r in values),
                assembly_successes=sum(r['modules']>1 and r['success']for r in values),groups=groups,
                failures=dict(Counter(r['reason']for r in values if not r['success'])),
                publication_max_s=max(pub,default=0.),publication_over_10ms=sum(x>.01 for x in pub),
                observed_interval_max_s=max(intervals,default=0.),observation_to_action_max_s=max(delays,default=0.),
                planning_max_s=max(r['planning_s']for r in values if 'planning_s'in r),
                successful_assembly_max_s=max((r['assembly_s']for r in values if r['success']),default=None),
                contrasts=ordered,issues=issues,checks=checks)
    (root/'audit.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();r=report(args.root)
    print(json.dumps({k:v for k,v in r.items()if k not in ('checks','contrasts')},indent=2))
