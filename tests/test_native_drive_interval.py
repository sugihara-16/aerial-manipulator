"""Native predictor and borrowed output views preserve physical derivatives."""
import numpy as np,pinocchio as pin,pytest
from amsrr_det.model import build_model
from amsrr_det.drive_prediction import DrivePrediction
from build.native_drive_interval_avx_fixed import NativeDriveInterval


@pytest.mark.parametrize('offset',[.001,.1])
def test_native_interval_matches_reference_and_keeps_previous_output(offset):
    model,meta=build_model(2,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2')])
    reference=DrivePrediction(model,meta);native=NativeDriveInterval(model,meta)
    rng=np.random.default_rng(1028);q=pin.integrate(model,pin.neutral(model),rng.normal(0,.005,model.nv))
    v=rng.normal(0,.001,model.nv);command=q[native.qi]+offset;force=np.full(len(meta['rotors']),4.)
    expected=reference.interval(q,v,command,force,period=.003,linearize=True,history=True)
    actual=native.interval(q,v,command,force,period=.003,linearize=True,history=True)
    for a,b in zip(expected[2]['samples'],actual[2]['samples']):
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-8,rtol=1e-9)
    held={key:value.copy()for key,value in actual[2]['samples'][-1].items()}
    native.interval(q,v,command+.001,force,period=.003,linearize=True,history=True)
    for key,value in held.items():np.testing.assert_array_equal(value,actual[2]['samples'][-1][key])
