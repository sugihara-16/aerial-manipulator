"""Audit saved search coverage separately from task success; no new simulation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def tree_key(modules, edges, surfaces=(), ports=False):
    """Tree isomorphism key, ignoring module numbering and the chosen root."""
    adjacent = [[] for _ in range(modules)]
    for edge in edges:
        a, b = edge['a'], edge['b']
        adjacent[a].append((b, edge['port_a'], edge['port_b']))
        adjacent[b].append((a, edge['port_b'], edge['port_a']))
    assert len(edges) == modules - 1
    def rooted(node, parent):
        children = [(pa if ports else '', pb if ports else '', rooted(other, node))
                    for other, pa, pb in adjacent[node] if other != parent]
        pads = tuple(sorted(port for module, port in surfaces if module == node)) if ports else ()
        return pads, tuple(sorted(children))
    return repr(min(rooted(node, -1) for node in range(modules)))


def audit(root):
    rows = []; hashes = {}; graph_counts = Counter(); port_counts = Counter()
    attempts = Counter(); family_counts = defaultdict(Counter)
    def read(path):
        raw = path.read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    for case_dir in sorted(p.parent for p in root.glob('*/result.json')):
        result = read(case_dir / 'result.json')
        request = read(case_dir / 'request.json')
        path = case_dir / 'generated_plan' / 'plan.json'
        if not path.exists():
            rows.append(dict(case=case_dir.name, plan_available=False, success=result['task_success']))
            continue
        plan = read(path); screens = read(case_dir / 'screens.json')
        model = plan['model']; surfaces = plan['surfaces']; history = plan['fit']['task_level_search']
        n = model['modules']; contacts = len(surfaces)
        graph = tree_key(n, model['edges'])
        port_graph = tree_key(n, model['edges'], surfaces, ports=True)
        graph_counts[graph] += 1; port_counts[port_graph] += 1
        family_counts[request['payload']['shape']][f'{n} modules/{contacts} contacts'] += 1
        for row in history:
            attempts[f"{row.get('modules')}/{row.get('contact_count')}/{row['status']}"] += 1
        rows.append(dict(case=case_dir.name, plan_available=True, success=result['task_success'],
            modules=n, contacts=contacts, edges=model['edges'], surfaces=surfaces,
            connectivity=graph, port_contact_graph=port_graph,
            grasp_axes=Counter(port.rstrip('12') for _, port in surfaces),
            search_rows=len(history), endpoint_candidates=len(screens),
            accepted_endpoint_candidates=sum(s.get('endpoint', {}).get('reason') == 'static_endpoint' for s in screens),
            four_contact_candidates=sum(r.get('contact_count') == 4 for r in history),
            selection_rule=plan['fit']['selection_rule'],
            orientation_unchanged=request['source'][3:] == request['destination'][3:]))
    planned = [r for r in rows if r['plan_available']]
    return dict(scope=__doc__, source=str(root), input_sha256=hashes,
        summary=dict(cases=len(rows), plans=len(planned), successes=sum(r['success'] for r in rows),
            module_contact_counts=Counter(f"{r['modules']} modules/{r['contacts']} contacts" for r in planned),
            connectivity_counts=graph_counts, port_contact_graph_counts=port_counts,
            top_three_port_contact_graph_cases=sum(sorted(port_counts.values(), reverse=True)[:3]),
            grasp_axes=sum((Counter(r['grasp_axes']) for r in planned), Counter()),
            search_row_counts=Counter(r['search_rows'] for r in planned),
            endpoint_candidate_counts=Counter(r['endpoint_candidates'] for r in planned),
            accepted_endpoint_candidate_counts=Counter(r['accepted_endpoint_candidates'] for r in planned),
            four_contact_candidate_cases=sum(r['four_contact_candidates'] > 0 for r in planned),
            recorded_attempt_counts=attempts, family_counts=family_counts,
            unchanged_orientation_cases=sum(r['orientation_unchanged'] for r in planned)), rows=rows,
        limitations=['Attempt logs count fitted/screened records, not every cheap enumerated graph.',
                    'Port-labelled tree keys ignore module numbering; physical reflection symmetries are not merged.',
                    'Coverage and physical reserve do not establish task success on unseen demands.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = audit(args.root)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result['summary'], indent=2))
