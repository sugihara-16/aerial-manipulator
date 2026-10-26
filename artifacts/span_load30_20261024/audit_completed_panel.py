"""Recheck recorded physical maxima and completion conditions, without rerunning tasks."""
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
panel = json.loads((root / 'panel.json').read_text())
issues = []
checked = []
limits = {
    'peak_contact_force_n': 30.00001,
    'peak_contact_moment_nm': 5.00001,
    'peak_penetration_m': .002,
    'peak_joint_speed_rad_s': 3.00001,
    'joint_limit_violation_rad': 1e-5,
    'peak_effort_ratio': 1.000001,
    'longest_carried_contact_gap_s': .05,
    'contact_material_displacement_m': .03,
}
for case in panel['cases']:
    path = root / case['id'] / 'result.json'
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    if not result.get('task_success'):
        continue
    errors = []
    audit = result['audit']
    for key, limit in limits.items():
        value = audit.get(key)
        if value is None or not math.isfinite(value) or value > limit:
            errors.append(key)
    for key in ('clearance_violations', 'unexpected_contacts'):
        if audit.get(key) != []:
            errors.append(key)
    if audit.get('minimum_transport_clearance_m', 0) < .1:
        errors.append('transport_clearance')
    if audit.get('time_since_gripper_contact_s', 0) < 2:
        errors.append('release_dwell')
    for key, limit in [('object_position_error_m', .03), ('object_orientation_error_rad', .2)]:
        if not math.isfinite(result[key]) or result[key] >= limit:
            errors.append(key)
    velocity = result['object_velocity']
    for name, values, limit in [('linear_speed', velocity[:3], .02), ('angular_speed', velocity[3:], .1)]:
        speed = math.sqrt(sum(v*v for v in values))
        if not math.isfinite(speed) or speed >= limit:
            errors.append(name)
    if result['reason'] != 'task_completed' or result['phase'] != 'RETREAT':
        errors.append('completion_phase')
    if errors:
        issues.append(dict(case=case['id'], errors=errors))
    checked.append(case['id'])
output = dict(scope='Independent recheck of saved maxima and final state; not a new simulation',
    successful_results_checked=len(checked), issues=issues, checked_cases=checked)
(root / 'completion_audit.json').write_text(json.dumps(output, indent=2))
print(json.dumps({key: value for key, value in output.items() if key != 'checked_cases'}))
