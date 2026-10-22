"""Check the lower bound against independently constructed feasible inputs."""
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from build.rotor_wrench_bound import rotor_plane_problem,wrench_input_lower_bound


def test_symmetric_hover_attains_the_mass_lower_bound():
    model,meta=build_model();q=pin.neutral(model)
    problem=rotor_plane_problem(model,meta,q)
    wrench=np.array([0.,0.,40.,0.,0.,0.])
    result=wrench_input_lower_bound(**dict(problem,middle=wrench,radius=np.zeros(6)))
    expected=4*10**1.5
    assert expected-1e-3<result['lower_bound']<=expected+1e-10


def test_dual_bound_never_exceeds_constructed_feasible_force_cost():
    model,meta=build_model();q=pin.neutral(model)
    problem=rotor_plane_problem(model,meta,q);maps=problem['maps']
    rng=np.random.default_rng(2904)
    for _ in range(30):
        angle=rng.uniform(-2.,2.,4);thrust=rng.uniform(.1,20.,4)
        force=thrust[:,None]*np.c_[np.cos(angle),np.sin(angle)]
        wrench=np.einsum('ijk,ik->j',maps,force)
        radius=rng.uniform(0.,.05,6)
        center=wrench+rng.uniform(-1.,1.,6)*radius
        result=wrench_input_lower_bound(**dict(problem,middle=center,radius=radius))
        assert result['lower_bound']<=sum(thrust**1.5)+1e-8


def test_gimbal_mass_interval_contains_exact_forward_kinematics():
    model,meta=build_model();q=pin.neutral(model)
    q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.8,-.5,.3)).coeffs()
    problem=rotor_plane_problem(model,meta,q);middle=problem['middle'];radius=problem['radius']
    data=model.createData();rng=np.random.default_rng(3301)
    moving=[a for a in meta['actuators'] if a['role']=='vectoring']
    for _ in range(60):
        for a in moving:q[a['q']]=rng.uniform(a['lower'],a['upper'])
        pin.framesForwardKinematics(model,data,q)
        origin=data.oMi[meta['bases'][0]['joint']].translation
        moment=np.zeros(3)
        for j,inertia in enumerate(model.inertias):
            position=data.oMi[j].act(inertia.lever)-origin
            moment+=np.cross(position,-inertia.mass*model.gravity.linear)
        assert np.all(abs(moment-middle[3:])<=radius[3:]+1e-10)
