"""Face restrictions affect synthesis and control, never collision physics."""
import json
from pathlib import Path
import mujoco
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.objects import ObjectShape
from amsrr_det.model import build_model
from amsrr_det import grasp_geometry as geometry
from amsrr_det import native_grasp_geometry as native
from amsrr_det.valve import ValveSpec,ValvePlant
from amsrr_det.valve_contacts import require_side_contacts,require_side_geometry
from amsrr_det.task_geometry import box_face_mask


def test_side_face_audit_cannot_accept_top_assisted_rotation():
    from amsrr_det.valve_audit import side_face_evidence
    rows=[dict(phase='TURN',non_side_normal_force=0.,non_side_axis_torque=0.)]
    assert side_face_evidence(rows)['passes']
    rows.append(dict(phase='TURN',non_side_normal_force=2.,non_side_axis_torque=.5))
    result=side_face_evidence(rows)
    assert not result['passes'] and result['violating_samples']==1
    assert not side_face_evidence([])['passes']


def test_face_identity_uses_position_and_excludes_ambiguous_edges():
    points=np.array([[.3,.025,.01],[.3,-.02499999,.015],[.3,0.,.025],[.45,0.,0.],[.3,.025,.025],[.6,.025,0.]])
    np.testing.assert_array_equal(box_face_mask(points,[.9,.05,.05],1),[True,True,False,False,False,False])


def test_projection_stays_on_designated_finite_face_when_top_is_closer():
    points=np.array([[.2,.02,.04],[.6,-.02,-.04]])
    normals=np.array([[0.,1.,0.],[0.,-1.,0.]])
    surface,jac=geometry.box_face_projection(points,np.array([.45,.025,.025]),normals)
    np.testing.assert_allclose(surface,[[.2,.025,.025],[.45,-.025,-.025]])
    np.testing.assert_array_equal(jac,[np.diag([0.,1.,1.]),np.eye(3)])


@pytest.mark.parametrize('extra_top',[False,True])
def test_whole_dock_face_check_includes_shallow_unintended_contacts(extra_top):
    top='<geom name="dock/rim" pos=".3 .03495 .03495" type="box" size=".01 .01 .01"/>' if extra_top else ''
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="object" type="box" size=".45 .025 .025"/>
      <body pos="0 -.03495 0"><freejoint/>
        <geom name="dock/anchor" pos=".3 0 0" type="box" size=".01 .01 .005"/>{top}
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    shape=ObjectShape('box',[.9,.05,.05],1.2)
    assert d.ncon>0 and min(c.dist for c in d.contact)>-.001
    if extra_top:
        with pytest.raises(ValueError,match='non-side'):
            require_side_geometry(m,d,shape)
    else:require_side_geometry(m,d,shape)


def test_edge_support_uses_normal_only_to_resolve_geometric_ties():
    points=np.array([[.3,-.02499998,.02500001],[.3,-.025,.025],
                     [.3,-.0247,.025],[.3,-.025,.01]])
    normals=np.array([[.02, .999, .001],[0.,0.,-1.],[0.,1.,0.],[.03,1.,.01]])
    np.testing.assert_array_equal(box_face_mask(points,[.9,.05,.05],1,normals),[True,False,False,True])


@pytest.mark.parametrize('top',[False,True])
@pytest.mark.parametrize('angle',[.01,.08,.2,.7,.78,.79,.9])
@pytest.mark.parametrize('depth',[.0001,.0005,.001])
def test_tilted_mesh_box_edge_has_correct_face_under_compression(top,angle,depth):
    import ctypes
    from amsrr_det.fast_task_audit import FastTaskAudit
    from test_relative_task_control import plant
    vertices=' '.join(str(v) for x in [-1,1] for y in [-1,1] for z in [-1,1] for v in [x,y,z])
    # A tilted rectangular pad presses the bar's y/z edge. Its compressive
    # normal is predominantly sideward; exchanging y and z makes a top push.
    cy=.025+.01/np.cos(angle)+np.tan(angle)*.025-depth/np.cos(angle)
    R=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    pos=np.array([.3,cy,0.])
    if top:
        swap=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        R=swap@R;pos=swap@pos
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel())
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0" cone="elliptic" ccd_tolerance="1e-12" ccd_iterations="1000"/>
      <asset><mesh name="pad" vertex="{vertices}" scale=".015 .01 .12"/></asset>
      <worldbody><body name="object"><geom name="object" type="box" size=".45 .025 .025"/></body>
      <body name="m0/pad" pos="{' '.join(map(str,pos))}" quat="{' '.join(map(str,quat))}"><freejoint/>
      <geom name="m0/pad/g" type="mesh" mesh="pad" condim="3"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);obj=m.geom('object').id
    assert d.ncon
    for c in d.contact:
        n=c.frame[:3]*(1 if c.geom[1]==obj else -1);point=c.pos+.5*c.dist*n
        expected=(angle<np.pi/4) != top
        assert bool(box_face_mask(point,[.9,.05,.05],1,n,c.dist)) == expected
    # The every-step C++ monitor uses the same geometry rule, independently
    # of the Python planner/controller and with actual solved contact forces.
    fixture=plant();fixture.mj.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    audit=FastTaskAudit(fixture,['pad'],[.1,.1,.1],0.)
    call=audit.library.audit_face_step;call.restype=None
    call.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    size=np.array([.9,.05,.05]);stats=np.zeros(15)
    call(m._address,d._address,obj,1,size.ctypes.data,stats.ctypes.data)
    assert bool(stats[1]) != expected


@pytest.mark.parametrize('top',[False,True])
@pytest.mark.parametrize('depth',[.00001,.0001,.0005,.001,.003])
@pytest.mark.parametrize('edge_gap',[.000001,.00001,.0001,.001,.01])
def test_face_interior_does_not_change_identity_with_depth(top,depth,edge_gap):
    # A flat pad entirely inside one finite bar face, progressively closer
    # to its edge. Depth cannot turn this into support by the other face.
    vertices=' '.join(str(v) for x in [-1,1] for y in [-1,1] for z in [-1,1] for v in [x,y,z])
    position=[.3,.035-depth,.025-.002-edge_gap];scale=[.01,.01,.002]
    if top:
        position[1],position[2]=position[2],position[1]
        scale[1],scale[2]=scale[2],scale[1]
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option ccd_tolerance="1e-12"/>
      <asset><mesh name="pad" vertex="{vertices}" scale="{' '.join(map(str,scale))}"/></asset>
      <worldbody><geom name="object" type="box" size=".45 .025 .025"/>
      <body pos="{' '.join(map(str,position))}"><freejoint/>
      <geom type="mesh" mesh="pad"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);obj=m.geom('object').id
    assert d.ncon
    for c in d.contact:
        n=c.frame[:3]*(1 if c.geom[1]==obj else -1);p=c.pos+.5*c.dist*n
        assert bool(box_face_mask(p,[.9,.05,.05],1,n,c.dist)) is not top


def test_penetration_envelope_never_admits_a_distant_or_top_dominated_face():
    points=np.array([[.6,.025,0.],[.3,.02427,.025],[.3,.0250009,.0249996]])
    normals=np.array([[.01,1.,0.],[.034,-.001,-.9994],[.034,-.001,-.9994]])
    assert not box_face_mask(points,[.9,.05,.05],1,normals,-.0006).any()


def test_material_fit_centers_the_short_side_without_changing_face():
    from amsrr_det.grasp_task import refine_material_posture
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_yaw_symmetry_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    q=np.asarray(saved['q']);surfaces=list(map(tuple,saved['surfaces']))
    data=p.pin.createData();pin.framesForwardKinematics(p.pin,data,q)
    R=data.oMf[p.meta['object']['frame']].rotation.copy();q[:3]+=.006*R[:,2]
    c=require_side_contacts(p,spec.shape,q,surfaces)
    points=np.array([np.mean([a['target_point'] for a in c],axis=0)])
    normals=np.array([-R.T@c[0]['normal']]);before=abs(points[0,2])
    oq=p.meta['object']['q']
    _,q,fit=refine_material_posture(p,spec.shape,q[oq:oq+7],q,surfaces,points,normals,
        open_posture=False,preserve_box_faces=True,face_center_axis=2)
    assert abs(points[0,2])<.001 and abs(points[0,2])<before
    require_side_contacts(p,spec.shape,q,surfaces)


@pytest.mark.parametrize('native_enabled',[False,True])
def test_fixed_face_fit_derivative_and_slide(monkeypatch,native_enabled):
    lib=native.library()
    if native_enabled and lib is None:pytest.skip('native kernel not built')
    monkeypatch.setattr(native,'library',lambda:lib if native_enabled else None)
    shape=ObjectShape('box',[.9,.05,.05],1.2)
    model,meta=build_model(1,[],shape.payload());q=pin.neutral(model)
    oq=meta['object']['q'];q[oq:oq+3]=[.2,-.1,1.5]
    R=pin.rpy.rpyToMatrix(.3,-.4,.5);q[oq+3:oq+7]=pin.Quaternion(R).coeffs()
    frame=meta['frames'][geometry.grasp_frame(0,'yaw1')]
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    # The frame begins closer to the top, although the assigned face is +y.
    q[:3]+=q[oq:oq+3]+R@np.array([.3,.021,.04])-data.oMf[frame].translation
    target=np.array([q[oq:oq+3]+R@np.array([.3,.025,0.])]);normal=np.array([-R[:,1]])
    solver=geometry.least_squares
    def inspect(fun,x0,*,jac,**kwargs):
        x=x0.copy();x[3:6]=[.1,-.12,.08]
        actual=jac(x);h=1e-6
        expected=np.column_stack([(fun(x+h*e)-fun(x-h*e))/(2*h) for e in np.eye(len(x))])
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-6)
        return solver(fun,x0,jac=jac,**kwargs)
    monkeypatch.setattr(geometry,'least_squares',inspect)
    final,report=geometry.fit_contacts(model,meta,q,[(0,'yaw1')],target,normal,
        base_rotation=True,object_shape=shape,preserve_box_faces=True,max_evaluations=80)
    assert report['preserved_box_faces']
    np.testing.assert_array_equal(report['inward_normals'],[[0.,-1.,0.]])
    assert report['position_error_m']<.001
    assert abs(report['object_points'][0][2])<=.025


def test_material_on_top_is_rejected_even_if_it_can_transmit_torque():
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_yaw_symmetry_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    q=np.asarray(saved['q']);surfaces=list(map(tuple,saved['surfaces']))
    require_side_contacts(p,spec.shape,q,surfaces)
    data=p.pin.createData();pin.framesForwardKinematics(p.pin,data,q)
    obj=data.oMf[p.meta['object']['frame']]
    turn=obj.rotation@pin.exp3(np.array([np.pi/2,0.,0.]))@obj.rotation.T
    q=q.copy();q[:3]=obj.translation+turn@(q[:3]-obj.translation)
    q[3:7]=pin.Quaternion(turn@pin.Quaternion(q[3:7]).matrix()).coeffs()
    with pytest.raises(ValueError,match='side faces'):
        require_side_contacts(p,spec.shape,q,surfaces)


def test_live_top_contact_is_not_a_force_variable_but_remains_physical():
    from build.relative_task_control import CurrentContactLoad,RobotDrive
    from test_relative_task_control import plant
    p=plant();m=p.mj
    m.geom_type[m.geom('object').id]=mujoco.mjtGeom.mjGEOM_BOX
    m.geom_size[m.geom('object').id]=.05
    shape=ObjectShape('box',[.1,.1,.1],2.)
    load=CurrentContactLoad(p,m.qpos0,['pad'],object_shape=shape,object_face_axis=0)
    drive=RobotDrive(p);q=m.qpos0.copy();q[0]=0.;q[2]=1.0999
    drive.observe(q,np.zeros(m.nv))
    c=load.prepare(drive,np.zeros(6),active_only=True)
    assert not c['active'].any()
    data=mujoco.MjData(m);data.qpos[:]=q;mujoco.mj_forward(m,data)
    assert data.ncon>0
    assert m.geom_contype[m.geom('object').id]!=0
