"""Check task forces and incremental kinematics independently of selection."""
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model, DockEdge, validate_graph
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.task_structure import grow_to_contacts
from amsrr_det.task_contacts import ContactRequirements


def request():
    return dict(payload=dict(shape='box',size=[.3,.2,.2],mass=.7),
                source=[0,0,.3,0,0,0,1],destination=[.3,0,.3,0,0,0,1],support_top=.2)


def test_contact_roles_must_balance_force_and_moment_for_all_task_vertices():
    requirements=ContactRequirements(request())
    p,n,_=requirements.choose(3,np.random.default_rng(4))
    ids=[next(i for i,x in enumerate(requirements.points) if np.allclose(x,point)) for point in p]
    assert requirements.cost(ids)[0]<1e-7
    assert len(ids)>=2
    assert requirements.cost(ids[:1])[0]>1e-3
    assert np.all(requirements.wrenches[:,3:]==0)
    np.testing.assert_allclose(requirements.wrenches[3,:3]+requirements.wrenches[4,:3],2*requirements.wrenches[0,:3])


def test_contact_force_margin_matches_symmetric_grasp_analytic_limit():
    task=request();task['destination']=task['source'].copy()
    requirements=ContactRequirements(task)
    ids=[int(np.argmin(np.linalg.norm(requirements.points-point,axis=1)))
         for point in [[-.15,0.,0.],[.15,0.,0.]]]
    residual,negative_margin=requirements.cost(ids)
    peak_tangent=np.max(requirements.wrenches[:,2])/2
    # The optimum equates the normal-force cap distance to the nearest
    # friction-plane distance. This independent force-space expression does
    # not use the ray coefficients of the implementation.
    normal_cap=30./5.5
    distance=(4.5*normal_cap-peak_tangent)/(4.5+np.sqrt(4.5**2+2))
    assert normal_cap-distance-.5>=distance
    assert residual<1e-7
    np.testing.assert_allclose(-negative_margin,distance/30.,atol=1e-8)


def test_contact_count_is_not_capped_by_nominal_two_point_feasibility():
    requirements=ContactRequirements(request())
    class FixedCount:
        def __init__(self,count):self.count=count;self.rng=np.random.default_rng(4)
        def integers(self,low,high):
            assert (low,high)==(1,9)
            return self.count
        def choice(self,*args,**kwargs):return self.rng.choice(*args,**kwargs)
    for count in [1,2,3,8]:
        points,_,report=requirements.choose(8,FixedCount(count))
        assert len(points)==count==report['requested_contacts']
        assert len(set(report['selected_sites']))==count
        assert report['normal_floor_N']==.5
        if count>=2:assert report['role_trace'][1]['residual']<1e-7


def test_partial_seams_match_independent_assembled_urdf_kinematics():
    points=np.array([[0,0,.5],[.6,0,.5]]);normals=np.array([[1.,0,0],[-1.,0,0]])
    for count in [2,3,5,8]:
        edges,surfaces,root,values,trace=grow_to_contacts(count,points,normals,np.random.default_rng(91))
        validate_graph(count,[DockEdge(**e) for e in edges])
        occupied={(e['a'],e['port_a']) for e in edges}|{(e['b'],e['port_b']) for e in edges}
        assert not occupied.intersection(surfaces)
        model,meta=build_model(count,edges);q=pin.neutral(model)
        q[:3]=root[:3,3];q[3:7]=pin.Quaternion(root[:3,:3]).coeffs()
        for a in meta['actuators']:
            module,name=a['name'].split('/');q[a['q']]=values[int(module[1:])].get(name,0.)
        data=model.createData();pin.framesForwardKinematics(model,data,q)
        first=data.oMf[meta['frames'][grasp_frame(*surfaces[0])]]
        np.testing.assert_allclose(first.translation,points[0],atol=1e-12)
        np.testing.assert_allclose(first.rotation[:,0],normals[0],atol=1e-12)
        last=data.oMf[meta['frames'][grasp_frame(*surfaces[-1])]]
        residual=np.sum((last.translation-points[-1])**2)+.15**2*np.sum((last.rotation[:,0]-normals[-1])**2)
        assert residual<=trace[-1]['residual_m2']+1e-12


def test_contact_separation_changes_discrete_production_with_same_randomness():
    normals=np.array([[1.,0,0],[-1.,0,0]])
    short=grow_to_contacts(3,[[0,0,.4],[.15,0,.4]],normals,np.random.default_rng(18))
    long=grow_to_contacts(3,[[0,0,.4],[1.,0,.4]],normals,np.random.default_rng(18))
    assert short[0]!=long[0]


def test_variable_count_auxiliary_attempts_three_role_candidates_below_eight(monkeypatch):
    import build.select_structure as selector
    seen=[]
    def receive(task,candidate,**kwargs):
        seen.append(candidate)
        return dict(accepted=False),None,None
    monkeypatch.setattr(selector,'screen_candidate',receive)
    # Regression input: the old prefix-only auxiliary attempted only 8 modules
    # when this supported, tilted ellipsoid generated three contact roles.
    rotation=[.594691222534657,.38254710592217184,.382547153044944,.5946912957896852]
    task=dict(payload=dict(shape='ellipsoid',size=[.5425451884452169,.29654530977126436,.401888844112138],mass=1.3648029275376143),
              source=[.3055855473015059,-.4050755622795362,.3110231253122171,*rotation],
              destination=[.6441751426743272,-.3448389470981015,.3110231253122171,*rotation],support_top=.1627504704265838)
    answer=selector.task_initial_candidate(task,False,20261023,starts=1)
    assert [c['modules'] for c in seen]==list(range(1,9))
    assert all(c['generation']['method']=='contact_directed_growth' for c in seen)
    assert any(c['modules']<8 and len(c['surfaces'])>=3 for c in seen)
    assert answer['prepared'] is None
    for c in seen:
        validate_graph(c['modules'],[DockEdge(**e) for e in c['edges']])


def test_opposed_pair_can_slide_and_its_compressive_forces_obey_friction():
    from amsrr_det.grasp_geometry import opposed_contact_terms
    from amsrr_det.objects import ObjectShape
    shape=ObjectShape('box',[.3,.4,.5],.7);mu=.6
    for shift in [0.,.06,-.07]:
        points=np.array([[-.15,shift,.02],[.15,shift+.03,-.01]])
        _,gap,outward=shape.project_surface(points)
        assert np.max(abs(gap))<1e-12
        assert not opposed_contact_terms(points,-outward,mu).any()
        force=(points[1]-points[0]);force/=np.linalg.norm(force)
        for f,n in zip([force,-force],-outward):
            compression=f@n
            assert compression>0 and np.linalg.norm(f-compression*n)<=mu*compression
        np.testing.assert_allclose(np.cross(points[0],force)+np.cross(points[1],-force),0.,atol=1e-12)
    assert opposed_contact_terms(np.array([[-.15,0.,0.],[-.15,.05,0.]]),np.array([[1.,0,0],[1.,0,0]]),mu).min()>0
