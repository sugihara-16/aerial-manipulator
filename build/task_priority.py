"""Only actuator output limits stop execution; other excursions are telemetry.

This policy does not change the simulator, actuator limits or final task goal.
Recorded contact force is not a calibrated structural-strength limit.
"""


def violation(audit):
    if audit.max_effort_ratio > 1.000001:
        return 'motor_effort_limit'
    return None


def secondary_violations(audit):
    values=dict(contact_force=audit.max_force > 30.00001,
                contact_moment=audit.max_moment > 5.00001,
                penetration=audit.max_penetration > .002,
                clearance=bool(audit.clearance_violations),
                contact_gap=audit.longest_gap > .05,
                material_slip=audit.max_material_displacement > .03,
                joint_speed=audit.max_joint_speed>3.00001,
                joint_range=audit.max_joint_limit_violation>1e-5,
                geometric_collision=audit.minimum_part_gap < -1e-5 or audit.minimum_environment_gap < -1e-5,
                unplanned_collision=bool(audit.other_contacts),
                transport_clearance=audit.min_clearance_carried<.10)
    return {key:bool(value) for key,value in values.items()}
