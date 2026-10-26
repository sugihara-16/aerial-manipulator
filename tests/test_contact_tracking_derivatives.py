"""World angular velocity and quaternion errors must share the same frame."""
import mujoco
import numpy as np
from build.held_contact_tracking import errors


def test_object_tracking_derivatives_match_perturbed_pose_and_velocity():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="object"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    rng=np.random.default_rng(723)
    for _ in range(20):
        target=model.qpos0.copy()
        mujoco.mj_integratePos(model,target,rng.normal(size=model.nv),.6)
        q=target.copy();mujoco.mj_integratePos(model,q,rng.normal(size=model.nv),.3)
        v=rng.normal(size=model.nv);velocity=rng.normal(size=model.nv)
        directions=rng.normal(size=(2*model.nv,4))
        _,derivatives=errors(model,q,v,target,velocity,directions)
        epsilon=1e-6
        for i in range(4):
            plus=q.copy();minus=q.copy()
            mujoco.mj_integratePos(model,plus,directions[:model.nv,i],epsilon)
            mujoco.mj_integratePos(model,minus,directions[:model.nv,i],-epsilon)
            measured=(errors(model,plus,v+epsilon*directions[model.nv:,i],target,velocity)
                      -errors(model,minus,v-epsilon*directions[model.nv:,i],target,velocity))/(2*epsilon)
            np.testing.assert_allclose(measured,derivatives[:,:,i],rtol=1e-7,atol=1e-8)


def test_prediction_wrapper_preserves_configured_contact_accuracy(monkeypatch):
    from types import SimpleNamespace
    from amsrr_det.mujoco_model import PlantModel
    from build.held_contact_tracking import HeldContactTracking,ResponseContactControl
    xml='''<mujoco><worldbody>
      <body name="pad"><joint name="hinge" range="-1 1"/>
        <geom name="pad/surface" type="sphere" size=".05" mass="1"/></body>
      <body pos=".11 0 0"><freejoint name="object"/>
        <geom name="object" type="sphere" size=".05" mass="1"/></body>
      </worldbody><actuator><general joint="hinge" gainprm="1" biasprm="0 -2 -1"
        forcerange="-5 5"/></actuator></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    meta={'actuators':[dict(name='hinge',role='dock',effort=5.)],'rotors':[]}
    plant=PlantModel(None,meta,model,xml,[])
    model.opt.ccd_iterations=1000;model.opt.ccd_tolerance=1e-12
    model.opt.timestep=.000125
    original=PlantModel.__init__
    def diagnostic_defaults(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.mj.opt.ccd_iterations=100;self.mj.opt.ccd_tolerance=1e-9
    monkeypatch.setattr(PlantModel,'__init__',diagnostic_defaults)
    observed=[]
    def check_model(self,*args,**kwargs):
        observed.append((self.m.opt.ccd_iterations,self.m.opt.ccd_tolerance))
        return kwargs['input_reference'].copy()
    monkeypatch.setattr(ResponseContactControl,'update',check_model)
    feedback=SimpleNamespace(m=model,local=SimpleNamespace(plant=plant,links=['pad']),
        nominal=model.qpos0.copy(),info={},clearance_model=model)
    held=HeldContactTracking(feedback)
    held.target=model.qpos0.copy();held.target_velocity=np.zeros(model.nv)
    held.reference=held.air_gaps.copy()
    try:
        held.correct(model.qpos0,np.zeros(model.nv),np.zeros(model.nu))
        assert observed==[(1000,1e-12)]
        assert held.model.opt.timestep==.0005 and model.opt.timestep==.000125
        assert plant.mj is model and held.optimizer.plant.mj is held.model
    finally:held.close()
