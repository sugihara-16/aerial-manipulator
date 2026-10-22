import json
import subprocess
import numpy as np
import pytest
from amsrr_det.model import export_model
from test_physics import execute,BINARY


def test_sdirk_freefall_has_second_order_position(tmp_path):
    model,meta=export_model(tmp_path)
    r=execute(tmp_path,model,meta,dict(duration_s=.2,integrator='drive_sdirk2',thrust_lag_s=0.),.002)
    assert r['full_horizon']
    assert r['final']['q'][2]==pytest.approx(1.+.5*model.gravity.linear[2]*.2**2,abs=1e-12)
    assert r['final']['v'][2]==pytest.approx(model.gravity.linear[2]*.2,abs=1e-12)
    assert r['counts']['rhs']>=4*r['counts']['steps']


def test_sdirk_thrust_transient_is_second_order(tmp_path):
    model,meta=export_model(tmp_path)
    scenario=dict(duration_s=.01,integrator='drive_sdirk2',thrust_lag_s=.03,command={'thrust':[3.]*4})
    exact=3*(1-np.exp(-.01/.03));errors=[]
    for h in [.002,.001,.0005]:
        r=execute(tmp_path,model,meta,scenario,h)
        assert r['full_horizon']
        errors.append(abs(r['final']['f'][0]-exact))
    assert errors[0]>3.8*errors[1]>3.8**2*errors[2]


def test_sdirk_contact_is_explicitly_unsupported(tmp_path):
    from test_physics import patch
    model,meta=export_model(tmp_path)
    source=tmp_path/'case.json'
    source.write_text(json.dumps(dict(duration_s=.1,integrator='drive_sdirk2',patches=[patch()])))
    r=subprocess.run([str(BINARY),str(tmp_path),str(source),'.002',str(tmp_path/'out.json')],capture_output=True,text=True)
    assert r.returncode==2
    assert 'excludes contact/history' in r.stderr
