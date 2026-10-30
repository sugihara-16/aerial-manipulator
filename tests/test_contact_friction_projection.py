"""Check the contact-law projection against an independent convex solve."""
import numpy as np
import scipy.sparse as sp
import clarabel
import pytest
from build.contact_friction_modes import project_force, G


def test_weighted_projection_matches_convex_force_problem():
    rng=np.random.default_rng(73)
    examples=[np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([.01,2.,-8.])]
    examples.extend(rng.normal(size=(24,3))*[.3,10.,10.])
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-11
    for ratio in [1.,10000.]:
        metric=np.array([138.,138./ratio,138./ratio])
        matrix=np.vstack([G,[-1.,0.,0.]])
        for affine in examples:
            mode,actual=project_force(affine,metric[0],ratio)
            answer=clarabel.DefaultSolver(sp.diags(metric,format='csc'),-metric*affine,
                sp.csc_matrix(matrix),np.zeros(9),[clarabel.NonnegativeConeT(9)],settings).solve()
            assert str(answer.status)in ['Solved','AlmostSolved']
            assert max(abs(actual-np.array(answer.x)))<2e-5
            assert max(matrix@actual)<1e-7


@pytest.mark.parametrize('fixture,separate_all',[
    ('release_zero_force_branch.npz',False),('release_full_separation_branch.npz',True)])
def test_release_can_leave_contact_without_attraction(fixture,separate_all):
    """Real releases previously remained in a feasible closed contact mode."""
    from pathlib import Path
    from build.contact_friction_modes import solve
    from build.held_input_solver import violation
    f=np.load(Path(__file__).parent/'fixtures'/fixture)
    nv,nu,nt=(int(f[k])for k in ['nv','nu','nt'])
    dims=f['cone_dims']
    cones=[clarabel.ZeroConeT(int(dims[0])),clarabel.NonnegativeConeT(int(dims[1])),
           *[clarabel.SecondOrderConeT(int(n))for n in dims[2:]]]
    x,correction,note=solve(f['P'],f['c'],f['A'],f['b'],cones,nv,nu,nt,
        f['pressure_matrix'],f['Dn'],f['p0'],f['rn'],float(f['ratio']),
        f['observed_velocity'],boundary_hint=f['hint'],separate_all=separate_all)
    cost=lambda z:.5*z@f['P']@z+f['c']@z
    assert cost(x)<cost(f['hint'])-.1
    apex=[i for i,mode in enumerate(note['friction_modes'])if mode=='apex']
    assert apex
    if separate_all:
        assert len(apex)==nt//2
        # Waiting for every force to reach numerical zero misses this branch.
        assert min(f['p0']-f['Dn']@f['hint'][:nv])>1e-5
    normal=f['p0']-f['Dn']@x[:nv]
    # The affine spring would pull; unilateral projection instead gives zero.
    assert min(normal[apex])<-.01
    np.testing.assert_allclose((normal+correction)[apex],0.,atol=1e-7)
    assert note['projection_error']<=1e-5
    # Tangential constitutive equations are replaced by the checked branch;
    # original dynamics, input limits and all force/moment cones remain.
    kept=np.r_[np.arange(nv),np.arange(nv+nt,len(f['b']))]
    revised=[clarabel.ZeroConeT(int(dims[0])-nt),*cones[1:]]
    assert violation(f['A'][kept],(f['b']-f['pressure_matrix']@correction)[kept],revised,x)<=1e-5


def test_preferred_branch_skips_proposal_solve_but_rechecks_force_law(monkeypatch):
    from pathlib import Path
    import build.contact_friction_modes as module
    f=np.load(Path(__file__).parent/'fixtures/release_full_separation_branch.npz')
    dims=f['cone_dims']
    kw={k:f[k]for k in ['P','c','A','b','pressure_matrix','Dn','p0','rn','observed_velocity']}
    kw.update(nv=int(f['nv']),nu=int(f['nu']),nt=int(f['nt']),ratio=float(f['ratio']),
        cones=[clarabel.ZeroConeT(int(dims[0])),clarabel.NonnegativeConeT(int(dims[1])),
               *[clarabel.SecondOrderConeT(int(n))for n in dims[2:]]],boundary_hint=f['hint'],separate_all=True)
    x,correction,note=module.solve(**kw)
    def unexpected(*args,**kwargs):raise AssertionError('cached valid branch need not solve a proposal')
    monkeypatch.setattr(module,'scaled_recovery',unexpected)
    count=0;project=module.project_forces
    def checked(*args):
        nonlocal count
        count+=len(args[0])
        return project(*args)
    monkeypatch.setattr(module,'project_forces',checked)
    again,shift,warm=module.solve(**kw,preferred_modes=note['friction_modes'])
    np.testing.assert_allclose(again,x,atol=1e-7)
    np.testing.assert_allclose(shift,correction,atol=1e-7)
    assert warm['preferred_mode_accepted'] and count==kw['nt']//2
    assert warm['projection_error']<=1e-5
    # A proposed closed branch must not be trusted merely because it is cached.
    monkeypatch.undo()
    _,_,retry=module.solve(**kw,preferred_modes=((),)*(kw['nt']//2))
    assert not retry['preferred_mode_accepted'] and retry['trials']>1
    assert retry['projection_error']<=1e-5


def test_optional_detachment_never_requires_slow_feasibility_recovery(monkeypatch):
    from pathlib import Path
    import build.contact_friction_modes as module
    f=np.load(Path(__file__).parent/'fixtures/release_full_separation_branch.npz')
    dims=f['cone_dims']
    def unexpected(*args,**kwargs):
        raise AssertionError('a feasible incumbent needs no slow recovery for optional improvement')
    monkeypatch.setattr(module,'scaled_recovery',unexpected)
    kw={k:f[k] for k in ['P','c','A','b','pressure_matrix','Dn','p0','rn','observed_velocity']}
    x,_,note=module.solve(**kw,nv=int(f['nv']),nu=int(f['nu']),nt=int(f['nt']),ratio=float(f['ratio']),
        cones=[clarabel.ZeroConeT(int(dims[0])),clarabel.NonnegativeConeT(int(dims[1])),
               *[clarabel.SecondOrderConeT(int(n)) for n in dims[2:]]],boundary_hint=f['hint'],separate_all=True)
    cost=lambda z:.5*z@f['P']@z+f['c']@z
    assert cost(x)<cost(f['hint'])-.1
    assert note['projection_error']<=1e-5
    assert set(note['friction_modes'])=={'apex'}


@pytest.mark.parametrize('fixture,fast',[
    ('observed_contact_branch',True),('observed_contact_retry',False)])
def test_observed_motion_branch_checks_physics_and_retains_recovery(monkeypatch,fixture,fast):
    """A real successful grasp formerly spent half a second proposing a branch."""
    import json
    from pathlib import Path
    import build.contact_friction_modes as module
    from build.held_input_solver import violation
    path=Path(__file__).parent/'fixtures'/(fixture+'.npz')
    meta=json.loads(path.with_suffix('.json').read_text());kw=dict(np.load(path))
    kw.update({k:None for k in meta['none']})
    kw['cones']=[getattr(clarabel,c['type'])(c['dim'])for c in meta['cones']]
    for k in ['nv','nu','nt']:kw[k]=int(kw[k])
    kw['ratio']=float(kw['ratio'])
    calls=[];original=module.scaled_recovery
    def recovery(*args,**kwargs):
        assert not fast,'valid observed branch needs no slow proposal'
        calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'scaled_recovery',recovery)
    x,correction,note=module.solve(**kw)
    assert note['observed_mode_accepted']==fast and not note['preferred_mode_accepted']
    assert bool(calls)==(not fast)
    assert note['projection_error']<=1e-5
    nv,nt=kw['nv'],kw['nt']
    retained=np.r_[np.arange(nv),np.arange(nv+nt,len(kw['b']))]
    cones=[clarabel.ZeroConeT(kw['cones'][0].dim-nt),*kw['cones'][1:]]
    assert violation(kw['A'][retained],(kw['b']-kw['pressure_matrix']@correction)[retained],cones,x)<=1e-7
