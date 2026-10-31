"""Local shared-angle refinement of already feasible torque endpoint trims.

One convex linearization proposes a common angle increment and separate force
vectors. The original nonlinear force screens alone accept the changed angles.
"""
import numpy as np
import scipy.sparse as sp
import clarabel
from .valve_trim import static_balance
from .grasp_task import material_force_contacts
from .structure_bounds import hover_input_burden


def refine(p,spec,q,surfaces,*,free_endpoint=False,steps=1,target=0.,initial=None):
    from .valve_plan import bearing_support
    contacts=material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
    actuators=[p.meta['actuators'][r['gimbal']] for r in p.meta['rotors']]
    qi=np.array([a['q'] for a in actuators]);nr=len(qi)
    options=dict(payload_fraction=0.,object_support=bearing_support(spec),
        minimum_pad_normal_force_n=.1,direct_burden=True)
    def screen(pose,models=False):
        reports=[];workspace={}
        for factor in ([0.,.9,1.1] if free_endpoint else [.9,1.1]):
            selected=dict(options)
            if factor==0.:selected.update(object_support=None,minimum_pad_normal_force_n=0.)
            report=static_balance(p.pin,p.meta,pose,() if factor==0. else contacts,workspace=workspace,direction_model=models,
                object_wrench=None if factor==0. else np.r_[np.zeros(3),np.asarray(spec.axis)*np.sign(spec.target_angle)*spec.torque*factor],**selected)
            if report['status']!='static_screen_pass':return None
            reports.append(report)
        return reports
    original=screen(q,True)
    if original is None:raise ValueError('direction refinement requires feasible inputs')
    cost=max(hover_input_burden(r['thrust']) for r in original)
    if initial is not None:
        proposed=q.copy();proposed[qi]=np.clip(np.asarray(initial)[qi],
            [a['lower']+.02 for a in actuators],[a['upper']-.02 for a in actuators])
        reports=screen(proposed,True)
        if reports is not None:
            improved=max(hover_input_burden(r['thrust']) for r in reports)
            if improved<cost-1e-6:q,original,cost=proposed,reports,improved
    def step(q,original,cost):
        sizes=[len(r['direction_model']['x']) for r in original]
        offsets=nr+np.r_[0,np.cumsum(sizes[:-1])];nv=nr+sum(sizes)
        for radius in [.05,.025,.0125]:
            rows=[];rhs=[];cones=[]
            for stage,report in enumerate(original):
                dm=report['direction_model'];start=offsets[stage];size=sizes[stage];K=dm['change']
                equality=np.zeros((len(dm['E']),nv));equality[:,start:start+size]=dm['E'];equality[:,:nr]=-K[dm['unactuated']]
                rows.append(equality);rhs.append(dm['equality']);cones.append(clarabel.ZeroConeT(len(equality)))
                matrix=np.zeros((len(dm['A']),nv));matrix[:,start:start+size]=dm['A']
                for i,a in enumerate(p.meta['actuators']):
                    matrix[2*i,:nr]=K[a['v']];matrix[2*i+1,:nr]=-K[a['v']]
                bounds=dm['bounds']+[(0.,1.)];extra=[];values=[]
                for k,(lo,hi) in enumerate(bounds):
                    for bound,sign in [(hi,1.),(lo,-1.)]:
                        if bound is None:continue
                        row=np.zeros(nv);row[start+k]=sign;extra.append(row);values.append(sign*bound)
                matrix=np.vstack([matrix,extra]);rows.append(matrix);rhs.append(np.r_[dm['b'],values]);cones.append(clarabel.NonnegativeConeT(len(matrix)))
            bounds=np.zeros((2*nr,nv));bounds[:nr,:nr]=np.eye(nr);bounds[nr:,:nr]=-np.eye(nr)
            rows.append(bounds);rhs.append(np.r_[np.minimum(radius,[a['upper']-.02-q[a['q']] for a in actuators]),
                -np.maximum(-radius,[a['lower']+.02-q[a['q']] for a in actuators])]);cones.append(clarabel.NonnegativeConeT(2*nr))
            # A quadratic proposal avoids power-cone epigraphs. Acceptance
            # still requires lower actual MAX sum(T**1.5) over both loads.
            objective=np.zeros(nv);H=np.zeros(nv);H[:nr]=.01
            for start in offsets:H[start:start+nr]=2.
            settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=60
            settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-8
            solved=clarabel.DefaultSolver(sp.diags(H,format='csc'),objective,sp.csc_matrix(np.vstack(rows)),np.concatenate(rhs),cones,settings).solve()
            if str(solved.status) not in ['Solved','AlmostSolved']:continue
            proposed=q.copy();proposed[qi]+=np.asarray(solved.x)[:nr]
            if np.any(proposed[qi]<[a['lower']+.02-1e-8 for a in actuators]) or np.any(proposed[qi]>[a['upper']-.02+1e-8 for a in actuators]):continue
            reports=screen(proposed,True)
            if reports is not None:
                chosen=max(reports,key=lambda r:hover_input_burden(r['thrust']))
                if hover_input_burden(chosen['thrust'])<cost-1e-6:return proposed,reports
        return q,original
    q=q.copy()
    for _ in range(steps):
        if cost<=target:break
        proposed,reports=step(q,original,cost)
        new_cost=max(hover_input_burden(r['thrust']) for r in reports)
        if new_cost>=cost-1e-6:break
        q,original,cost=proposed,reports,new_cost
    chosen=max(original,key=lambda r:hover_input_burden(r['thrust']));chosen.pop('direction_model',None)
    return q.copy(),chosen
