import os,time
import numpy as np
from build.async_allocation import AsyncAllocation
class Slow:
 def __init__(self):self.last_command=np.array([1.]);self.last={}
 def update(self,*a,**k):
  time.sleep(.04);self.last={'completed':True};return np.array([2.])
def test_slow_allocation_cannot_block_or_arrive_before_measured_time():
 a=AsyncAllocation(Slow(),min(os.sched_getaffinity(0)))
 try:
  start=time.perf_counter();assert a.update(0.,'TEST',(),{})[0]==1.
  assert time.perf_counter()-start<.01
  time.sleep(.06) # receive is ready in wall time, but not yet in plant time
  assert a.update(.01,'TEST',(),{})[0]==1.
  assert a.records[0]['available_s']>=.04
  available=a.records[0]['available_s']
  assert a.update(available-1e-6,'TEST',(),{})[0]==1.
  assert a.update(available+1e-6,'TEST',(),{})[0]==2.
  assert a.records[0]['application_tick_s']>=available
 finally:a.close()

class SlowSend:
 def __init__(self):self.last_command=np.array([1.]);self.last={}
 def update(self,*a,**k):
  import socket
  original=socket.socket.send
  def delay(sock,data,*args,**kwargs):
   time.sleep(.025)
   return original(sock,data,*args,**kwargs)
  socket.socket.send=delay
  return np.array([2.])
def test_result_send_delay_is_included_before_physical_availability():
 a=AsyncAllocation(SlowSend(),min(os.sched_getaffinity(0)))
 try:
  a.update(0.,'TEST',(),{})
  time.sleep(.09)
  assert a.update(.01,'TEST',(),{})[0]==1.
  assert a.records[0]['response_transport_s']>=.025
  assert a.records[0]['available_s']>=.025
  assert a.update(a.records[0]['available_s']+.001,'TEST',(),{})[0]==2.
 finally:a.close()

def test_reference_preparation_time_is_not_erased_from_observation_age():
 a=AsyncAllocation(Slow(),min(os.sched_getaffinity(0)))
 try:
  started=time.perf_counter()
  time.sleep(.025)
  a.update(0.,'TEST',(),{},observed_wall=started)
  time.sleep(.08)
  assert a.update(.05,'TEST',(),{})[0]==1.
  r=a.records[0]
  assert r['dispatch_delay_s']>=.025
  assert r['available_s']>=.065
  assert abs(r['calculation_s']-sum(r[k]for k in ['dispatch_delay_s','queue_delay_s','worker_calculation_s','response_transport_s']))<1e-8
 finally:a.close()

class Quick:
 def __init__(self):self.last_command=np.array([1.]);self.last={}
 def update(self,*a,**k):return np.array([2.])
def test_publish_between_observation_ticks_without_early_arrival():
 a=AsyncAllocation(Quick(),min(os.sched_getaffinity(0)))
 try:
  a.update(0.,'TEST',(),{});time.sleep(.02)
  assert not a.poll(0.)
  available=a.records[0]['available_s']
  assert available>0.
  assert not a.poll(available-1e-6)
  assert a.poll(available+1e-6)
  assert a.command[0]==2.
  assert a.job is None
 finally:a.close()
