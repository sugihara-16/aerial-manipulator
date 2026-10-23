"""Summarize one complete, unchanged panel; never combine experiments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_grasp_panel import report
from scripts.report_structure_selection import report as structures


def stats(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else dict(
        count=len(values), minimum=float(values.min()),
        median=float(np.median(values)), p95=float(np.quantile(values, .95)),
        maximum=float(values.max()))


def summarize(root):
    panel_path = root/'panel.json'
    panel = json.loads(panel_path.read_text())
    scored = report(panel, [root], panel['configuration'])
    if scored['missing']:
        raise ValueError('The complete panel is required')
    detailed = []
    phase_time = defaultdict(list)
    phases = Counter()
    for row in scored['rows']:
        result = json.loads(Path(row['result_path']).read_text())
        detailed.append(result)
        if not row['success']:
            phases[(result.get('phase'), row['reason'])] += 1
        for phase, times in result.get('phase_control_time_s', {}).items():
            phase_time[phase].append(times)
    selection = structures(root)
    (root/'structure_report.json').write_text(json.dumps(selection, indent=2))
    timing = json.loads((root/'timing.json').read_text())
    planning = [r['initial_planning_s'] for r in detailed]
    result = dict(
        scope='One implementation on all 120 specified inputs; these inputs have been used in development, not an unseen holdout',
        panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        overall=scored['overall'], missing=scored['missing'],
        same_runtime_sources=scored['all_results_same_runtime_sources'],
        same_execution_settings=scored['all_results_same_execution_settings'],
        all_cases_cover_required_sequence=scored['all_cases_cover_required_sequence'],
        strata=scored['strata'], failure_concentration=scored['failure_concentration'],
        failure_phases=[dict(phase=p, reason=r, count=c) for (p, r), c in phases.items()],
        planning_s=stats(planning),
        planning_over_15s=sum(t > 15. for t in planning),
        planning_successful_s=stats([r['initial_planning_s'] for r in detailed if r['task_success']]),
        control_timing_scope='Distribution across cases of each case mean/p95/maximum; not pooled cycle percentiles',
        control_s={phase: {metric: stats([v[metric] for v in values])
                           for metric in ['mean', 'p95', 'maximum']}
                   for phase, values in phase_time.items()},
        task_wall_to_simulation_ratio=stats([r['wall_s']/r['simulation_s'] for r in detailed if r.get('simulation_s', 0) > 0]),
        benchmark_wall_s=timing['wall_s'],
        benchmark_workers=timing['workers'],
        structure={k: v for k, v in selection.items() if k != 'rows'},
        runtime_platform='PC; no VIM4 or physical robot measurement',
    )
    (root/'validation_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['overall', 'same_runtime_sources', 'same_execution_settings', 'planning_s', 'planning_over_15s', 'benchmark_wall_s']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
