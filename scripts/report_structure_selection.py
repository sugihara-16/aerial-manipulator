"""Separate sampled, physically accepted, selected and successfully executed structures."""
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path


def signature(modules,edges,surfaces=(),ports=True):
    neighbors=defaultdict(list);pads=defaultdict(list)
    for edge in edges:
        a,b=edge['a'],edge['b'];pa,pb=edge['port_a'],edge['port_b']
        neighbors[a].append((b,pa,pb));neighbors[b].append((a,pb,pa))
    for module,port in surfaces:pads[module].append(port)
    def rooted(node,parent):
        children=[(pa,pb,rooted(other,node))if ports else rooted(other,node)
                  for other,pa,pb in neighbors[node]if other!=parent]
        return (tuple(sorted(pads[node])),tuple(sorted(children)))if ports else tuple(sorted(children))
    return min(rooted(node,None)for node in range(modules))


def report(root):
    rows=[];sampled=Counter();accepted=Counter();sampled_contacts=Counter();accepted_contacts=Counter()
    sampled_groups=set();selected=Counter();executed=Counter();successful=Counter();skeletons=set();port_structures=set();used_pads=Counter()
    for path in sorted(root.glob('*/selection.json')):
        case=path.parent.name;selection=json.loads(path.read_text())
        result=json.loads((path.parent/'result.json').read_text())
        for trial in selection.get('rows',[]):
            for start in trial['rows']:
                sampled[trial['modules']]+=1;sampled_contacts[len(start['surfaces'])]+=1;sampled_groups.add(start['group'])
                if start['accepted']:accepted[trial['modules']]+=1;accepted_contacts[len(start['surfaces'])]+=1
        evidence=selection.get('selection_evidence')or{}
        cost=selection.get('selected_burden');initial=(selection.get('initial')or{}).get('burden')
        better=cost is not None and initial is not None and cost<initial-1e-8*(1+initial)
        row=dict(case=case,selected_burden=cost,initial_seed_burden=initial,
                 improves_initial_seed=better,improvement_fraction=(initial-cost)/initial if better else 0.,
                 selection_evidence_complete=evidence.get('complete',False),seed=selection.get('seed'),
                 seed_source=selection.get('seed_source'),task_success=result.get('task_success',False),
                 initial_planning_s=result.get('initial_planning_s'),reason=result.get('reason'),
                 count_evidence=evidence.get('groups'))
        plan_path=path.parent/'generated_plan/plan.json'
        if plan_path.exists():
            plan=json.loads(plan_path.read_text());n=plan['model']['modules'];edges=plan['model']['edges'];surfaces=plan['surfaces']
            key=f'{n} modules / {len(surfaces)} contacts';selected[key]+=1
            if result.get('simulation_s',0)>0:executed[key]+=1
            if result.get('task_success'):successful[key]+=1
            skeletons.add(signature(n,edges,ports=False));port_structures.add(signature(n,edges,surfaces))
            for _,port in surfaces:used_pads['pitch'if port.startswith('pitch')else'yaw']+=1
            degree=Counter(node for e in edges for node in[e['a'],e['b']])
            row.update(modules=n,contacts=len(surfaces),branched=max(degree.values(),default=0)>2,
                       edges=edges,surfaces=surfaces)
        rows.append(row)
    return dict(scope='finite completed panel; sampled counts are not uniform coverage or physical success probabilities',
        cases=len(rows),all_selection_evidence_complete=all(r['selection_evidence_complete']for r in rows),
        fresh_seed_cases=sum(r['seed_source']=='fresh_system_entropy'for r in rows),
        unique_seeds=len({r['seed']for r in rows}),improved_initial_seed_cases=sum(r['improves_initial_seed']for r in rows),
        sampled_starts_by_count=dict(sampled),accepted_starts_by_count=dict(accepted),
        sampled_contact_counts=dict(sampled_contacts),accepted_contact_counts=dict(accepted_contacts),
        sampled_skeleton_groups=sorted(sampled_groups),selected=dict(selected),executed=dict(executed),successful=dict(successful),
        selected_unlabeled_skeletons=len(skeletons),selected_port_contact_structures=len(port_structures),
        selected_pad_types=dict(used_pads),selected_branched_cases=sum(r.get('branched',False)for r in rows),rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=report(args.root);(args.root/'structure_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items()if key!='rows'},indent=2))
