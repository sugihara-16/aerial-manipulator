"""A real thin EPA face must satisfy the unchanged contact accuracy checks."""
from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import pytest
from amsrr_det.mujoco_model import make_plant, coat_grippers, contact_dynamics_model, CONTACT_CACHE
from build.mit_drive import dock_mit_model


@pytest.mark.parametrize("fixture_name", ["epa_thin_face.json", "epa_thin_plane.json", "epa_nonconvex_seed.json"])
def test_thin_face_witness_does_not_abort_contact_generation(fixture_name):
    fixture = json.loads((Path(__file__).parent/'fixtures'/fixture_name).read_text())
    plant = make_plant(fixture['modules'], fixture['edges'], fixture['payload'],
                       mesh_cache=CONTACT_CACHE, floor=True)
    links = [f'm{a}/'+('yaw_dock_mech' if b.startswith('yaw') else 'pitch_dock_mech')+b[-1]
             for a, b in fixture['surfaces']]
    doc = ET.fromstring(plant.xml)
    doc.find('option').set('integrator', 'discrete')
    for attributes in fixture['supports']:
        ET.SubElement(doc.find('worldbody'), 'geom', **attributes)
    xml = coat_grippers(ET.tostring(doc, encoding='unicode'), links, reference_model=plant.mj)
    plant = replace(plant, xml=xml, mj=mujoco.MjModel.from_xml_string(xml))
    model = contact_dynamics_model(dock_mit_model(plant, kp=200.), links).mj
    model.opt.ccd_tolerance = 1e-12
    model.opt.ccd_iterations = 1000
    data = mujoco.MjData(model)
    data.qpos[:] = fixture['q']
    mujoco.mj_forward(model, data)
    ids = {model.geom(name).id for name in fixture['pair']}
    contacts = [c for c in data.contact if {int(c.geom1), int(c.geom2)} == ids]
    assert contacts
    assert abs(min(c.dist for c in contacts)-fixture['expected_distance']) < 5e-11
