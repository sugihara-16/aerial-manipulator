"""Summarize saved task outcomes without changing the test denominator."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scripts.report_grasp_panel import report
from scripts.report_structure_selection import report as structure_report


def distribution(values):
    if not values:
        return None
    return dict(count=len(values), minimum=min(values), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=max(values))


def summarize(root):
    panel = json.loads((root / 'panel.json').read_text())
    evaluation = report(panel, [root], panel['configuration'])
    structure = structure_report(root)
    initial, initial_success, execution_ratios, retained_durations = [], [], [], []
    phase_means, phase_maxima = defaultdict(list), defaultdict(list)
    failures = Counter()
    successes_without_complete_evidence = []
    selected_counts = Counter()
    for row in evaluation['rows']:
        if row['result_path'] is None:
            continue
        path = Path(row['result_path'])
        result = json.loads(path.read_text())
        if result.get('initial_planning_s') is not None:
            initial.append(result['initial_planning_s'])
            if row['success']:
                initial_success.append(result['initial_planning_s'])
        if result.get('simulation_s', 0) > 0:
            execution_ratios.append(result['wall_s'] / result['simulation_s'])
        for phase, times in result.get('phase_control_time_s', {}).items():
            phase_means[phase].append(times['mean'])
            phase_maxima[phase].append(times['maximum'])
        if row['success']:
            selection = json.loads((path.parent / 'selection.json').read_text())
            evidence = selection.get('selection_evidence') or {}
            if not evidence.get('complete'):
                successes_without_complete_evidence.append(row['case'])
            selected_counts[result['modules']] += 1
            duration = result.get('audit', {}).get('time_since_gripper_contact_s')
            if duration is not None:
                retained_durations.append(duration)
        else:
            failures[(str(result.get('phase')), row['reason'])] += 1
    return dict(
        evaluation=evaluation, structure=structure,
        timing=dict(initial_all_s=distribution(initial), initial_successes_s=distribution(initial_success),
                    initial_over_15_s=sum(x > 15 for x in initial),
                    audited_execution_wall_over_simulation=distribution(execution_ratios),
                    phase_case_mean_s={p: distribution(v) for p, v in phase_means.items()},
                    phase_worst_call_s={p: max(v) for p, v in phase_maxima.items()},
                    phase_scope='Distribution of per-case mean control times; physical integration and independent audit excluded'),
        successes_without_complete_selection_evidence=successes_without_complete_evidence,
        successful_module_counts=dict(selected_counts),
        failures_by_phase=[dict(phase=p, reason=r, count=n) for (p, r), n in failures.items()],
        time_since_first_pad_contact_at_success_s=distribution(retained_durations),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    (args.root / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(overall=summary['evaluation']['overall'],
                          ready=summary['evaluation']['ready_for_completion_review'],
                          timing=summary['timing'],
                          success_counts=summary['successful_module_counts'],
                          missing_selection_evidence=summary['successes_without_complete_selection_evidence']), indent=2))
