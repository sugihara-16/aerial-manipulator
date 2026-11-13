"""Nonblocking state publication; rendering never runs in the control process."""
import json
import mmap
import struct
import time
from pathlib import Path
import numpy as np

HEADER=128


class LiveState:
    def __init__(self,folder,*,preparation_started=None):
        self.folder=Path(folder);self.map=None;self.next_wall=0.;self.counter=0;self.ready_wall=None
        self.preparation_started=preparation_started
        self.phase=''

    def begin(self,model,xml):
        self.folder.mkdir(parents=True,exist_ok=True)
        (self.folder/'live.xml').write_text(xml)
        self.nq,self.nv,self.ne=model.nq,model.nv,model.neq
        size=HEADER+8*(self.nq+self.nv+self.ne)
        self.file=open(self.folder/'live.bin','w+b');self.file.truncate(size)
        self.map=mmap.mmap(self.file.fileno(),size)
        self.payload=np.ndarray((self.nq+self.nv+self.ne,),dtype=np.float64,buffer=self.map,offset=HEADER)
        (self.folder/'live.json').write_text(json.dumps(dict(nq=self.nq,nv=self.nv,neq=self.ne,bytes=size)))

    def ready(self):
        if self.ready_wall is None:
            self.ready_wall=time.perf_counter()
            if self.preparation_started is not None:
                (self.folder/'ready.json').write_text(json.dumps(dict(preparation_s=self.ready_wall-self.preparation_started)))

    def update(self,data,phase):
        self.phase=phase
        now=time.monotonic()
        if self.map is None or now<self.next_wall:return
        self.next_wall=now+1/60
        self.counter+=2
        struct.pack_into('Q',self.map,0,self.counter-1)
        self.payload[:self.nq]=data.qpos
        self.payload[self.nq:self.nq+self.nv]=data.qvel
        self.payload[self.nq+self.nv:]=data.eq_active
        struct.pack_into('d',self.map,8,float(data.time))
        label=str(phase).encode('utf8')[:80];self.map[16:96]=label.ljust(80,b'\0')
        struct.pack_into('Q',self.map,0,self.counter)

    def close(self):
        if self.map is not None:
            del self.payload;self.map.close();self.file.close();self.map=None


def snapshot(mapped,meta):
    a=struct.unpack_from('Q',mapped,0)[0]
    if not a or a%2:return None
    t=struct.unpack_from('d',mapped,8)[0];phase=mapped[16:96].split(b'\0')[0].decode('utf8')
    values=np.frombuffer(mapped,dtype=np.float64,offset=HEADER).copy()
    if a!=struct.unpack_from('Q',mapped,0)[0]:return None
    return t,phase,values
