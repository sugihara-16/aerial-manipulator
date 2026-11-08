"""GUI sampling must preserve generation and tolerate display failure."""
import json
import numpy as np
from amsrr_det.planning_preview import GrowthPreview, read_previews
from amsrr_det.task_structure import grow_to_contacts


def test_disabled_preview_performs_no_output(tmp_path, monkeypatch):
    monkeypatch.delenv('AMSRR_PLANNING_PREVIEW', raising=False)
    GrowthPreview(1).update(None, None, None, None)
    assert not list(tmp_path.iterdir())


def test_preview_preserves_generated_tree_and_random_state(tmp_path, monkeypatch):
    args=(4, [[0,0,.4],[.4,0,.4]], [[1,0,0],[-1,0,0]])
    monkeypatch.delenv('AMSRR_PLANNING_PREVIEW', raising=False)
    rng=np.random.default_rng(91);expected=grow_to_contacts(*args,rng);next_draw=rng.random()
    monkeypatch.setenv('AMSRR_PLANNING_PREVIEW',str(tmp_path))
    rng=np.random.default_rng(91);actual=grow_to_contacts(*args,rng)
    assert rng.random()==next_draw
    encode=lambda v:json.dumps(v,default=lambda x:x.tolist(),sort_keys=True)
    assert encode(actual)==encode(expected)
    rows=read_previews(tmp_path)
    assert len(rows)==1 and rows[0]['built']==4
    assert rows[0]['edges']==actual[0]
    assert rows[0]['stage']=='topology_generated'
    assert len(rows[0]['poses'])==4


def test_latest_complete_snapshot_per_count(tmp_path):
    for name,n,t in [('a',3,1),('b',3,3),('c',4,2)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(dict(modules=n,timestamp_ns=t,candidate=name)))
    (tmp_path/'incomplete.json').write_text('{')
    assert [r['candidate'] for r in read_previews(tmp_path)]==['b','c']


def test_display_write_failure_does_not_break_generation(tmp_path, monkeypatch):
    monkeypatch.setenv('AMSRR_PLANNING_PREVIEW',str(tmp_path/'missing'))
    preview=GrowthPreview(1)
    preview.update([np.eye(4)],[],[],np.zeros((1,3)))
    assert preview.folder is None


def test_both_walking_families_publish_without_changing_output(tmp_path, monkeypatch):
    from amsrr_det.walking import WalkSpec
    from amsrr_det.walking_growth import radial_generate, shared_generate
    def run(generator):
        rng=np.random.default_rng(12)
        try:
            result=generator(WalkSpec(modules=4,speed=.22),rng)
            result=json.dumps(result,default=lambda x:x.tolist(),sort_keys=True)
        except ValueError as e:
            result=str(e)
        return result,rng.random()
    for generator in [radial_generate,shared_generate]:
        monkeypatch.delenv('AMSRR_PLANNING_PREVIEW',raising=False)
        expected=run(generator)
        folder=tmp_path/generator.__name__;folder.mkdir()
        monkeypatch.setenv('AMSRR_PLANNING_PREVIEW',str(folder))
        assert run(generator)==expected
        assert read_previews(folder)[0]['modules']==4


def test_checked_and_selected_candidates_are_not_replaced_by_later_growth(tmp_path,monkeypatch):
    import pinocchio as pin
    from amsrr_det.model import build_model
    from amsrr_det.planning_preview import SearchPreview,compact_candidate,select_candidate
    monkeypatch.setenv('AMSRR_PLANNING_PREVIEW',str(tmp_path))
    model,_=build_model();q=pin.neutral(model);q[:3]=[1.,2.,3.]
    candidate=lambda:compact_candidate(1,[],q,[(0,'yaw1')])
    preview=SearchPreview();preview.outcome(1,'accepted',2.,candidate)
    preview.outcome(1,'rejected');preview.outcome(1,'bounded_nonimprovement')
    preview.finish();select_candidate(candidate)
    GrowthPreview(1).update([np.eye(4)],[],[],np.zeros((1,3)))
    row=read_previews(tmp_path)[0]
    assert row['checks']==dict(checked=3,accepted=1,rejected=1,bounded=1,cancelled=0,best_cost=2.)
    assert row['selected'] and row['search_finished']
    np.testing.assert_allclose(np.array(row['validated']['poses'])[0,:3,3],q[:3])
    np.testing.assert_allclose(np.array(row['poses'])[0,:3,3],0.)


def test_only_improved_checked_geometry_is_requested(tmp_path,monkeypatch):
    from amsrr_det.planning_preview import SearchPreview
    monkeypatch.setenv('AMSRR_PLANNING_PREVIEW',str(tmp_path))
    calls=[];preview=SearchPreview()
    def candidate():calls.append(1);return dict(example=True)
    preview.outcome(3,'accepted',10.,candidate)
    preview.outcome(3,'accepted',12.,candidate)
    preview.outcome(3,'accepted',9.,candidate)
    preview.finish()
    assert len(calls)==2
    assert json.loads((tmp_path/'checks.json').read_text())['rows']['3']['accepted']==3


def test_flat_display_uses_zero_joints_and_coincident_connection_ports():
    from amsrr_det.planning_preview import flat_geometry
    from amsrr_det.model import build_model
    import pinocchio as pin
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),dict(a=1,port_a='pitch2',b=2,port_b='yaw2')]
    view=flat_geometry(3,json.dumps(edges,sort_keys=True))
    model,meta=build_model(3,edges);data=model.createData()
    pin.framesForwardKinematics(model,data,pin.neutral(model))
    for i in range(3):
        np.testing.assert_allclose(view['poses'][i],data.oMf[meta['frames'][f'm{i}/root']].homogeneous)
        np.testing.assert_allclose(np.array(view['poses'][i])[:3,2],[0,0,1],atol=1e-12)
        assert len(view['rotors'][i])==4
    for edge in edges:
        np.testing.assert_allclose(view['ports'][edge['a']][edge['port_a']],view['ports'][edge['b']][edge['port_b']],atol=1e-12)
