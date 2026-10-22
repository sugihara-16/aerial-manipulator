"""Physical graph/target invariants independent of the search ranking."""
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.expanded_search import contact_extensions,additional_sites
from amsrr_det.model import DockEdge,validate_graph
from amsrr_det.grasp_geometry import free_surfaces
from amsrr_det.objects import ObjectShape


@pytest.mark.parametrize('limit',[3,4,5,8])
def test_growth_keeps_assigned_pads_free_and_obeys_real_port_rules(limit):
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),dict(a=0,port_a='pitch2',b=2,port_b='yaw2')]
    pads=[(1,'yaw2'),(2,'yaw1')];rows=list(contact_extensions(3,edges,pads,limit))
    assert rows
    for n,graph,newpad in rows:
        assert n<=limit and len(graph)==n-1
        validate_graph(n,[DockEdge(**e) for e in graph])
        assert set(pads+[newpad])<=set(free_surfaces(dict(modules=n,edges=graph)))
        assert newpad not in pads
    if limit>3:
        assert {e[2]['a'] for n,e,_ in rows if n>3}=={0,1,2}
        assert {pad[1][:-1] for _,_,pad in rows}=={'pitch','yaw'}


@pytest.mark.parametrize('family',['box','cylinder','ellipsoid','triangular_prism','tapered_box'])
def test_new_targets_are_distinct_real_object_surfaces_under_rotation(family):
    shape=ObjectShape(family,[.3,.3,.2],.6)
    pose=np.r_[.2,-.3,.4,pin.Quaternion(pin.rpy.rpyToMatrix(.4,-.3,.7)).coeffs()]
    points,_=shape.opposed_sites(pose,np.array([1.,0.,0.]))
    sites=additional_sites(shape,pose,points)
    assert len(sites)>=2
    for point,inward in sites:
        projected,gap,_=shape.project_surface([point-1e-5*inward])
        np.testing.assert_allclose(projected[0],point,atol=1e-8)
        np.testing.assert_allclose(gap,[1e-5],atol=1e-8)
        assert min(np.linalg.norm(points-point,axis=1))>=.02
        assert np.linalg.norm(inward)==pytest.approx(1.)


@pytest.mark.parametrize('limit',[3,4,5,8])
@pytest.mark.parametrize('contacts',[2,3,4])
def test_branch_lengthening_preserves_contact_count_and_other_free_pad(limit,contacts):
    from amsrr_det.expanded_search import contact_replacements
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),dict(a=0,port_a='pitch2',b=2,port_b='yaw2')]
    pads=[(1,'yaw2'),(2,'yaw1'),(0,'yaw1'),(0,'yaw2')][:contacts]
    rows=list(contact_replacements(3,edges,pads,limit))
    assert bool(rows)==(limit>3)
    for n,graph,newpad,index in rows:
        validate_graph(n,[DockEdge(**e)for e in graph]);assert n<=limit
        selected=pads.copy();selected[index]=newpad
        assert len(set(selected))==contacts
        free=set(free_surfaces(dict(modules=n,edges=graph)))
        assert set(selected)<=free and pads[index] not in free
        assert graph[:2]==edges
    if rows:
        assert {row[3]for row in rows}==set(range(contacts))
        assert {row[2][1][:-1]for row in rows}=={'pitch','yaw'}

@pytest.mark.parametrize('family',['box','cylinder','ellipsoid','triangular_prism','tapered_box'])
@pytest.mark.parametrize('tilt',[-np.pi/6,np.pi/6])
def test_inclined_side_sites_follow_world_height_and_real_surface(family,tilt):
    shape=ObjectShape(family,[.3,.3,.2],.6)
    pose=np.r_[.2,-.3,.4,pin.Quaternion(pin.rpy.rpyToMatrix(0.,0.,.7)).coeffs()]
    points,normals=shape.opposed_sites(pose,np.array([1.,0.,0.]),tilt)
    R=pin.Quaternion(pose[3:]).matrix()
    assert np.all(np.sign((points@R.T)[:,2])==np.sign(tilt))
    for point,inward in zip(points,normals):
        projected,gap,_=shape.project_surface([point-1e-5*inward])
        np.testing.assert_allclose(projected[0],point,atol=1e-8)
        np.testing.assert_allclose(gap,[1e-5],atol=1e-8)
        assert np.linalg.norm(inward)==pytest.approx(1.)


def test_additional_sites_include_lower_sides_without_removing_upper_sites():
    shape=ObjectShape('ellipsoid',[.3,.3,.2],.6);pose=np.array([0.,0.,.4,0.,0.,0.,1.])
    points,_=shape.opposed_sites(pose,np.array([1.,0.,0.]))
    sites=additional_sites(shape,pose,points)
    assert any(p[2]<0 and np.linalg.norm(p[:2])>1e-3 for p,_ in sites)
    assert any(p[2]>0 for p,_ in sites)


@pytest.mark.parametrize('family',['box','cylinder','ellipsoid','triangular_prism','tapered_box'])
@pytest.mark.parametrize('rpy',[(0.,0.,.7),(.4,-.3,.7)])
def test_lower_side_seed_preserves_table_clearance_under_object_rotation(family,rpy):
    from amsrr_det.structure_search import side_inclinations
    shape=ObjectShape(family,[.3,.3,.2],.6)
    pose=np.r_[.2,-.3,0.,pin.Quaternion(pin.rpy.rpyToMatrix(*rpy)).coeffs()]
    table=.3;pose[2]=table-shape.lowest_z(pose)
    axis=np.array([1.,0.,0.]);angles=side_inclinations(shape,pose,axis,table)
    assert angles[0]==0. and angles[1]<0. and angles[2]>0.
    points,_=shape.opposed_sites(pose,axis,angles[1]);R=pin.Quaternion(pose[3:]).matrix()
    height=(points@R.T)[:,2]+pose[2]
    assert np.min(height)>=table+.03-1e-7
    assert np.max(height)<pose[2]


def test_candidate_scheduler_visits_modes_and_counts_before_lengthening(monkeypatch):
    """Growing more arms must not block the other grasp axis."""
    from types import SimpleNamespace
    from amsrr_det.structure_search import grasp_candidates
    import amsrr_det.expanded_search as search
    visited=[]
    def postures(*args,contact_counts,mode,**kwargs):
        count,=contact_counts
        for extension in range(20):
            visited.append((count,mode,extension))
            plan=SimpleNamespace(fit={},wall_s=0.,count=count)
            yield dict(plant=SimpleNamespace(meta=dict(modules=3,edges=[])),
                       surfaces=[(i,'yaw1')for i in range(count)],fit={},plan=plan)
    monkeypatch.setattr(search,'expanded_postures',postures)
    shape=ObjectShape('box',[.3,.3,.2],.6)
    source=np.array([0.,0.,.4,0.,0.,0.,1.]);destination=source.copy();destination[0]+=.3
    candidates=grasp_candidates(shape,source,destination,.2)
    assert [next(candidates).count for _ in range(6)]==[2,2,3,3,4,4]
    assert visited==[(count,mode,0) for count in [2,3,4] for mode in ['yaw','pitch']]


@pytest.mark.parametrize('four_only',[False,True])
def test_four_contacts_can_start_from_one_parent_without_refitting_it(monkeypatch,four_only):
    from types import SimpleNamespace
    import amsrr_det.expanded_search as search
    fits=[]
    def draft(count):
        return dict(plant=SimpleNamespace(meta=dict(modules=count+1,edges=[])),closed=np.zeros(7),
            surfaces=[(i,'yaw1')for i in range(count)],points=np.zeros((count,3)),normals=np.zeros((count,3)),
            fit=dict(position_error_m=0.,normal_error=0.),score=0.)
    monkeypatch.setattr(search,'plan_seed',lambda *a,**kw:draft(2))
    monkeypatch.setattr(search,'finish_expanded_posture',lambda d,*a:SimpleNamespace(
        plant=d['plant'],closed_q=d['closed'],surfaces=d['surfaces'],points=d['points'],normals=d['normals']))
    def extend(parent,*a,**kw):
        count=len(parent['surfaces'])+1
        for index in range(12):
            fits.append((count,index));yield draft(count)
    monkeypatch.setattr(search,'extend_posture',extend)
    args=(None,None,None,0.)
    if four_only:
        stream=search.expanded_postures(*args,contact_counts=(4,),screen_parents=True)
    else:
        shared={}
        three=search.expanded_postures(*args,contact_counts=(3,),screen_parents=True,seed_state=shared)
        stream=search.expanded_postures(*args,contact_counts=(4,),screen_parents=True,seed_state=shared)
        assert len(next(three)['surfaces'])==3
    assert len(next(stream)['surfaces'])==4
    assert fits==[(3,0),(4,0)]


def test_four_contact_growth_does_not_require_three_contact_load_feasibility(monkeypatch):
    """The assembled final robot can use a parent that cannot carry the load."""
    from types import SimpleNamespace
    import amsrr_det.expanded_search as search
    def draft(count):
        return dict(plant=SimpleNamespace(meta=dict(modules=count+1,edges=[])),closed=np.zeros(7),
            surfaces=[(i,'yaw1')for i in range(count)],points=np.zeros((count,3)),normals=np.zeros((count,3)),
            fit=dict(position_error_m=0.,normal_error=0.),score=0.)
    monkeypatch.setattr(search,'plan_seed',lambda *a,**kw:draft(2))
    def screen(candidate,*args):
        if len(candidate['surfaces'])<4:raise ValueError('load screen failed')
        return SimpleNamespace(plant=candidate['plant'],closed_q=candidate['closed'],
            surfaces=candidate['surfaces'],points=candidate['points'],normals=candidate['normals'])
    monkeypatch.setattr(search,'finish_expanded_posture',screen)
    visited=[]
    def extend(parent,*a,**kw):
        count=len(parent['surfaces'])+1;visited.append(count)
        yield draft(count)
    monkeypatch.setattr(search,'extend_posture',extend)
    shared={};args=(None,None,None,0.)
    three=search.expanded_postures(*args,contact_counts=(3,),screen_parents=True,seed_state=shared)
    rejected=next(three)
    assert rejected['rejection']=='load screen failed'
    four=search.expanded_postures(*args,contact_counts=(4,),screen_parents=True,seed_state=shared)
    child=next(four)
    assert len(child['plan'].surfaces)==4 and visited==[3,4]
