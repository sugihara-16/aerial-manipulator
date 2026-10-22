"""Physical invariants for the four-node pad, independent of task control."""
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import export_model
from test_contact_integration import object_landing
from test_physics import execute


def supported_box(tmp_path,mu=.6):
    model,meta=export_model(tmp_path,payload=dict(mass=1.,size=[.3,.4,.15]))
    scene=object_landing(model,meta)
    scene['q0'][meta['object']['q']+2]=.075-9.80665/7500
    scene['v0']=[0.]*model.nv
    scene['patches'][0].update(mu=mu,ct=0.)
    scene['duration_s']=.1
    return model,meta,scene


@pytest.mark.parametrize('direction',[[1.,0.,0.],[0.,1.,0.],[.6,.8,0.]])
def test_coulomb_force_obeys_weight_and_direction(tmp_path,direction):
    model,meta,scene=supported_box(tmp_path)
    # A large elastic tangential displacement saturates the pad. Its net force
    # is mu*m*g irrespective of how many nodes represent the contact area.
    scene.update(probe=True,xi0=(np.tile(direction,4)*.1).tolist())
    result=execute(tmp_path,model,meta,scene,.0005)
    iv=meta['object']['v']
    assert np.allclose(result['acceleration'][iv:iv+3],-.6*9.80665*np.array(direction),atol=1e-10)


def object_energy(meta,scene,q,v,xi):
    iq,iv=meta['object']['q'],meta['object']['v']
    orientation=pin.Quaternion(*[q[iq+6],*q[iq+3:iq+6]]).matrix()
    # Box inertia about its COM; free-flyer velocities are body expressed.
    inertia=np.array([.4**2+.15**2,.3**2+.15**2,.3**2+.4**2])/12
    energy=9.80665*q[iq+2]+.5*np.dot(v[iv:iv+3],v[iv:iv+3])+.5*np.dot(inertia,np.square(v[iv+3:iv+6]))
    patch=scene['patches'][0]
    for i,point in enumerate(patch['points']):
        gap=(np.array(q[iq:iq+3])+orientation@point)[2]
        if gap<0:
            energy+=.5*patch['kn']/4*gap*gap+.5*patch['kt']/4*np.dot(xi[3*i:3*i+3],xi[3*i:3*i+3])
    return energy


def test_friction_sliding_dissipates_energy_and_converges(tmp_path):
    model,meta,scene=supported_box(tmp_path)
    scene['v0'][meta['object']['v']]=.15
    results=[]
    initial=object_energy(meta,scene,scene['q0'],scene['v0'],np.zeros(12))
    for h in [.001,.0005,.00025]:
        r=execute(tmp_path,model,meta,scene,h)
        assert r['full_horizon']
        state=r['final'];final=object_energy(meta,scene,state['q'],state['v'],state['xi'])
        assert final<initial-1e-4
        assert r['metrics']['peak_penetration_m']<.002
        results.append(np.array(state['q'])[meta['object']['q']:])
    assert np.max(abs(results[2]-results[1]))<np.max(abs(results[1]-results[0]))


def test_separation_removes_pad_memory(tmp_path):
    model,meta,scene=supported_box(tmp_path)
    scene['q0'][meta['object']['q']+2]=.2
    scene.update(xi0=[.1,0.,0.]*4,duration_s=.002)
    r=execute(tmp_path,model,meta,scene,.001)
    assert r['full_horizon']
    assert r['final']['xi']==[0.]*12
    assert r['metrics']['peak_node_force_n']==0.


def test_patch_limit_applies_to_sum_of_four_nodes(tmp_path):
    model,meta,scene=supported_box(tmp_path)
    scene['patches'][0]['max_force_n']=5.
    r=execute(tmp_path,model,meta,scene,.0005)
    assert r['metrics']['peak_node_force_n']<5.
    assert r['metrics']['peak_patch_force_n']>9.
    assert r['stop_reason']=='physical_limit_violation'
    assert not r['full_horizon'] and not r['task_accepted']


def test_opposing_internal_load_is_rejected_despite_zero_object_acceleration(tmp_path):
    model,meta,scene=supported_box(tmp_path,mu=0.)
    scene['patches'][0]['max_force_n']=20.
    for sign in [-1,1]:
        scene['patches'].append(dict(frame='object',
            points=[[sign*.15,y,z] for y,z in [(-.01,-.01),(-.01,.01),(.01,-.01),(.01,.01)]],
            normal=[-sign,0.,0.],offset=-.149,kn=7500.,cn=75.,kt=7500.,ct=0.,mu=0.,
            lower=[-10.]*3,upper=[10.]*3,max_force_n=5.))
    r=execute(tmp_path,model,meta,scene,.0005)
    # Both side patches apply 7.5 N, canceling in the object's net wrench.
    # Each node is below 5 N and the object remains stationary, yet both patch
    # limits are violated. A net-wrench or motion-only gate would miss this.
    np.testing.assert_allclose(r['final']['v'][meta['object']['v']:],0.,atol=1e-11)
    assert r['metrics']['peak_node_force_n']<5.
    assert r['metrics']['max_contact_load_ratio']==pytest.approx(1.5,abs=1e-11)
    assert r['stop_reason']=='physical_limit_violation'
    assert not r['task_accepted']
