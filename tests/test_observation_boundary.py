import json
import subprocess
from amsrr_det.diagnostics import fixture
from test_physics import execute,BINARY


def test_hidden_initial_thrust_does_not_change_first_controller_command(tmp_path):
    model,meta,scene=fixture(tmp_path)
    scene.update(duration_s=.001,integrator='drive_implicit_euler')
    first=execute(tmp_path,model,meta,scene,.001,closed=True)
    scene['f0']=[x+.1 for x in scene['f0']]
    second=execute(tmp_path,model,meta,scene,.001,closed=True)
    assert first['samples'][0]['thrust_command']==second['samples'][0]['thrust_command']
    assert first['samples'][0]['position_command']==second['samples'][0]['position_command']
    assert first['final']['v']!=second['final']['v']


def test_missing_thrust_history_is_not_filled_with_simulator_truth(tmp_path):
    model,meta,scene=fixture(tmp_path)
    del scene['initial_thrust_estimate']
    source=tmp_path/'case.json';source.write_text(json.dumps(scene))
    r=subprocess.run([str(BINARY),str(tmp_path),str(source),'.002',str(tmp_path/'out.json'),'closed'],capture_output=True,text=True)
    assert r.returncode==2
    assert 'hidden plant state is not an observation' in r.stderr


def test_delayed_sensor_requires_declared_initial_packet(tmp_path):
    model,meta,scene=fixture(tmp_path)
    scene.update(observation_delay_s=.02,duration_s=.05,integrator='drive_implicit_euler')
    source=tmp_path/'delay.json';source.write_text(json.dumps(scene))
    r=subprocess.run([str(BINARY),str(tmp_path),str(source),'.002',str(tmp_path/'out.json'),'closed'],capture_output=True,text=True)
    assert r.returncode==2
    assert 'requires initial observation history' in r.stderr
    scene['initial_observation']=dict(timestamp=-.02,q=scene['q0'],v=[0.]*model.nv,motor_effort=scene['tau0'])
    result=execute(tmp_path,model,meta,scene,.002,closed=True)
    assert result['full_horizon']
    assert result['counts']['control']==5
    assert not result['task_accepted']


def test_malformed_initial_observation_is_rejected_before_dynamics(tmp_path):
    model,meta,scene=fixture(tmp_path)
    scene['initial_observation']=dict(timestamp=0.,q=scene['q0'],v=[0.],motor_effort=scene['tau0'])
    source=tmp_path/'bad_packet.json';source.write_text(json.dumps(scene))
    r=subprocess.run([str(BINARY),str(tmp_path),str(source),'.002',str(tmp_path/'out.json'),'closed'],capture_output=True,text=True)
    assert r.returncode==2
    assert 'invalid initial observation' in r.stderr
