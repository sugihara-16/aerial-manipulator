"""Diagnostic native contact geometry with explicit numerical accuracy settings."""
import hashlib,json,os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from amsrr_det.mujoco_model import PlantModel
from build.run_mujoco_task import main

original_init=PlantModel.__init__
def precise_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.mj.opt.ccd_tolerance=1e-9;self.mj.opt.ccd_iterations=100
    doc=ET.fromstring(self.xml);doc.find('option').set('ccd_tolerance','1e-9');doc.find('option').set('ccd_iterations','100')
    self.xml=ET.tostring(doc,encoding='unicode')
PlantModel.__init__=precise_init

