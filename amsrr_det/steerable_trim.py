"""Static direction proposal using sampled rotor-force rays.

Convex ray coefficients are a proposal, not extra physical actuators. Decode
one real angle per rotor, then run the unchanged physical static screen.
The sampled convex problem is not claimed to be an inner model of every
allowed one-axis sector; a sector wider than pi need not contain the result.
"""
import numpy as np
from .rotor_distribution import rotor_action_basis,rotor_basis_actions
from .equilibrium import static_balance


def steerable_trim(model,meta,q,contacts=(),**settings):
    from . import planning_memo
    if planning_memo.active is None:return _steerable_trim(model,meta,q,contacts,**settings)
    def plain(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return value
    return planning_memo.evaluate(plain(dict(operation='steerable_trim',modules=meta['modules'],
        edges=meta['edges'],actuators=meta['actuators'],object=meta['object'],gravity=model.gravity.linear,
        q=np.asarray(q),contacts=contacts,settings=settings)),
        lambda:_steerable_trim(model,meta,q,contacts,**settings))


def _steerable_trim(model,meta,q,contacts=(),**settings):
    actuators=[meta['actuators'][r['gimbal']] for r in meta['rotors']]
    qi=np.array([a['q'] for a in actuators]);q=q.copy()
    angles=np.linspace([a['lower']+.02 for a in actuators],
                       [a['upper']-.02 for a in actuators],9)
    rays=[];basis=rotor_action_basis(model,meta,q)
    for row in angles:
        rays.append(rotor_basis_actions(basis,row))
    proposal_settings={k:v for k,v in settings.items() if k!='direct_burden'}
    for _ in range(3):
        preference=(1.+1-np.cos(angles-q[qi])).T
        proposal=static_balance(model,meta,q,contacts,rotor_rays=np.asarray(rays),rotor_direction_cost=preference,**proposal_settings)
        if proposal['status']!='direction_proposal':return q,proposal
        coefficients=np.asarray(proposal['rotor_coefficients']).T
        vertical=np.sum(coefficients*np.cos(angles),axis=0)
        tangent=np.sum(coefficients*np.sin(angles),axis=0)
        # A zero-force rotor has no defined force direction. Preserve its
        # current angle instead of inventing atan2(0,0)=0 and a path jump.
        active=np.hypot(vertical,tangent)>1e-8
        decoded=np.clip(np.arctan2(tangent,vertical),angles[0],angles[-1])
        q[qi[active]]=decoded[active]
        result=static_balance(model,meta,q,contacts,**settings)
        result['direction_proposal_peak']=proposal['minimum_peak_effort_ratio']
        if result['status']=='static_screen_pass':return q,result
    return q,result
