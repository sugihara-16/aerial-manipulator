import numpy as np
from amsrr_det.valve import ValvePlant,ValveSpec
from amsrr_det.rotor_direction import RotorDirection
from build.relative_task_control import RobotDrive


def test_joint_direction_sensitivity_matches_independent_physical_columns():
    system=ValvePlant(1,[],ValveSpec(),meshes=False)
    dr=RobotDrive(system.control);q=dr.m.qpos0.copy();v=np.zeros(dr.m.nv)
    g=np.array([r['gimbal'] for r in system.control.meta['rotors']])
    q[dr.qi[g]]=[.2,-.3,.4,-.1];dr.observe(q,v)
    direction=RotorDirection(dr);force=np.array([3.,4.,5.,6.])
    B,bias=direction.linearize(q,v,force);baseline=dr.B.copy()
    holding=dr.d.qfrc_bias[dr.vi[g]]-baseline[np.ix_(dr.vi[g],np.arange(dr.nj,dr.m.nu))]@force
    inertia=np.diag(dr.mass)[dr.vi[g]];kd=dr.kd[g];kp=dr.kp[g];h=direction.horizon
    from scipy.linalg import expm
    gains=np.array([(1-expm(np.array([[0,1],[-k/m,-d/m]])*h)[0,0])/k for m,d,k in zip(inertia,kd,kp)])
    for j,a in enumerate(g):
        columns=[]
        for sign in [-1,1]:
            probe=q.copy();probe[dr.qi[a]]+=sign*1e-6;dr.observe(probe,v)
            columns.append(dr.B[:,dr.nj+j].copy())
        expected=(columns[1]-columns[0])/(2e-6)*force[j]*gains[j]
        np.testing.assert_allclose(B[:,a]-baseline[:,a],expected,atol=1e-8)
    # The effort supporting gravity/current thrust must not predict motion.
    np.testing.assert_allclose(bias+(B-baseline)[:,g]@holding,0.,atol=1e-12)
    np.testing.assert_array_equal(B[dr.ov:dr.ov+6],0.)
