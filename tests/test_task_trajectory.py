import numpy as np

from amsrr_det.grasp_task import smooth,quintic_path_velocity,placement_height


def test_nominal_world_velocity_matches_moving_position_goal():
    displacement=np.array([-.3,.4,.12]);duration=12.;h=1e-5
    for elapsed in [-1.,0.,2.,6.,11.,12.,14.]:
        numerical=displacement*(smooth((elapsed+h)/duration)-smooth((elapsed-h)/duration))/(2*h)
        np.testing.assert_allclose(quintic_path_velocity(displacement,elapsed,duration),
                                   np.r_[numerical,np.zeros(3)],atol=1e-10)


def test_placement_velocity_matches_corrected_goal_without_observed_velocity():
    h=1e-5
    for extent in [-.01,0.,.01]:
        for t in [0.,3.,6.,8.,9.,10.,12.]:
            goal,velocity=placement_height(.12,t,extent,.0001)
            old_height=.12*(1-smooth(t/10))
            assert abs(goal-(old_height+smooth((.02-old_height)/.02)*(extent+.0001)))<1e-14
            numerical=(placement_height(.12,t+h,extent,.0001)[0]-placement_height(.12,t-h,extent,.0001)[0])/(2*h)
            assert abs(velocity-numerical)<1e-10
