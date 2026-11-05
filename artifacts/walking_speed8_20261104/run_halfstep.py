"""Numerical-step check; retain the controller and measured-delay execution."""
import runpy,sys
sys.argv.extend(['--timestep','.00025'])
runpy.run_module('scripts.run_walking_task',run_name='__main__')
