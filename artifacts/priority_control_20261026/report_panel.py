"""Independently check the saved task/drive limits and summarize one full panel."""
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
previous = Path('artifacts/runs/span_load30_v11_20261026')
panel = json.loads((root / 'panel.json').read_text())
analysis = json.loads((root / 'analysis.json').read_text())
issues, rows, checked = [], [], []
secondary_counts = collections.Counter()
for case in panel['cases']:
    folder = root / case['id']
    result = json.loads((folder / 'result.json').read_text())
    before = json.loads((previous / case['id'] / 'result.json').read_text())
    errors = []
    settings = result['execution_configuration']
    if not settings.get('task_priority_control'):
        errors.append('task_priority_disabled')
    if settings.get('contact_update_period_s') != .01 or settings.get('steps') != 80:
        errors.append('time_grid_changed')
    if json.loads((folder / 'request.json').read_text()) != case:
        errors.append('input_changed')
    for name, value in result.get('secondary_safety_violations', {}).items():
        secondary_counts[name] += bool(value)
    if result.get('task_success'):
        checked.append(case['id'])
        audit = result['audit']
        for key, limit in {'peak_joint_speed_rad_s': 3.00001,
                           'joint_limit_violation_rad': 1e-5,
                           'peak_effort_ratio': 1.000001}.items():
            value = audit.get(key)
            if value is None or not math.isfinite(value) or value > limit:
                errors.append(key)
        if audit.get('unexpected_contacts') != []:
            errors.append('unexpected_contacts')
        for key in ['minimum_part_gap_capped_m', 'minimum_environment_gap_capped_m']:
            value = audit.get(key)
            if value is None or not math.isfinite(value) or value < -1e-5:
                errors.append(key)
        for key, limit in [('minimum_transport_clearance_m', .1), ('time_since_gripper_contact_s', 2.)]:
            value = audit.get(key)
            if value is None or not math.isfinite(value) or value < limit:
                errors.append(key)
        for key, limit in [('object_position_error_m', .03), ('object_orientation_error_rad', .2)]:
            if not math.isfinite(result[key]) or result[key] >= limit:
                errors.append(key)
        for name, values, limit in [('linear_speed', result['object_velocity'][:3], .02),
                                     ('angular_speed', result['object_velocity'][3:], .1)]:
            speed = math.sqrt(sum(v*v for v in values))
            if not math.isfinite(speed) or speed >= limit:
                errors.append(name)
        if result.get('reason') != 'task_completed' or result.get('phase') != 'RETREAT':
            errors.append('completion_phase')
    if errors:
        issues.append(dict(case=case['id'], errors=errors))
    plan = folder / 'generated_plan/plan.json'
    old_plan = previous / case['id'] / 'generated_plan/plan.json'
    same_structure = None
    if plan.exists() and old_plan.exists():
        a, b = json.loads(plan.read_text()), json.loads(old_plan.read_text())
        same_structure = (a['model']['edges'] == b['model']['edges'] and
                          a['model']['modules'] == b['model']['modules'] and
                          a['surfaces'] == b['surfaces'])
    rows.append(dict(case=case['id'], modules=result.get('modules'), success=result.get('task_success', False),
        reason=result.get('reason'), phase=result.get('phase'), planning_s=result.get('initial_planning_s'),
        execution_wall_s=result.get('wall_s'), simulation_s=result.get('simulation_s'),
        control_time_s=result.get('control_time_s'), phase_control_time_s=result.get('phase_control_time_s'),
        secondary_safety_violations=result.get('secondary_safety_violations'),
        old_success=before.get('task_success', False), old_reason=before.get('reason'),
        old_execution_wall_s=before.get('wall_s'), same_connection_and_pad_structure=same_structure))

completed = dict(scope='Recheck of saved task/drive maxima and final state; auxiliary safety values are recorded separately',
                 successful_results_checked=len(checked), issues=issues, checked_cases=checked)
(root / 'completion_audit.json').write_text(json.dumps(completed, indent=2))
summary = dict(scope='Fresh generation and full execution of the fixed 30 development cases, task-priority control',
    overall=analysis['overall'], all_30_results=len(rows) == 30,
    all_results_same_runtime_sources=analysis['all_results_same_runtime_sources'],
    all_results_same_execution_settings=analysis['all_results_same_execution_settings'],
    input_unchanged=panel == json.loads((previous / 'panel.json').read_text()),
    seeds_unchanged=json.loads((root / 'trial_seeds.json').read_text()) == json.loads((previous / 'trial_seeds.json').read_text()),
    original_input_sha256=hashlib.sha256(Path('artifacts/runs/span_load30_inputs_20261024.json').read_bytes()).hexdigest(),
    planning_seconds=analysis['planning_seconds'], module_counts=analysis['module_counts'],
    success_module_counts=analysis['success_module_counts'], complete_selection_evidence=analysis['complete_selection_evidence'],
    secondary_safety_violation_case_counts=dict(secondary_counts), completion_audit=completed,
    timing=json.loads((root / 'timing.json').read_text()), rows=rows)
(root / 'final_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ['overall', 'all_30_results', 'input_unchanged', 'seeds_unchanged',
    'all_results_same_runtime_sources', 'all_results_same_execution_settings', 'module_counts',
    'success_module_counts', 'planning_seconds', 'secondary_safety_violation_case_counts']}))
print(json.dumps(completed))
if issues:
    raise SystemExit(1)
