"""Object wrench requirements select contact roles before robot construction.

The small force-cone program ignores the robot. Its residual is a proposal
priority, not a claim that an assembled robot can realize those forces.
"""
from functools import lru_cache
import hashlib
import json
import numpy as np
import pinocchio as pin
from .objects import ObjectShape
from .planning_memo import evaluate as memoized


def _role_program(A,b,scenarios):
    """Solve independent force scenarios with the same shared priorities.

    Each scenario has its own forces. The common minimum residual is the
    largest individual minimum. At that residual, the common force margin is
    the smallest individual maximum. This is the same product feasible set,
    with one small HiGHS model whose basis is reused for changed task forces.
    """
    from scipy.sparse import csc_matrix
    from scipy.optimize._highspy import _core as highs
    rows,width=A.shape;count=(width-2)//scenarios;nr=rows//scenarios
    block=np.c_[A[:nr,:count],A[:nr,-2:]];sparse=csc_matrix(block);size=count+2
    lp=highs.HighsLp();lp.num_col_=size;lp.num_row_=nr
    lp.a_matrix_.num_col_=size;lp.a_matrix_.num_row_=nr;lp.a_matrix_.format_=highs.MatrixFormat.kColwise
    lp.a_matrix_.start_=sparse.indptr;lp.a_matrix_.index_=sparse.indices;lp.a_matrix_.value_=sparse.data
    objective=np.zeros(size);objective[-2]=1.
    lp.col_cost_=objective;lp.col_lower_=np.zeros(size);lp.col_upper_=np.r_[np.full(size-1,np.inf),0.]
    lp.row_lower_=np.full(nr,-np.inf);lp.row_upper_=b[:nr]
    solver=highs._Highs();options=highs.HighsOptions();options.output_flag=False;options.threads=1
    solver.passOptions(options);solver.passModel(lp)
    residual=0.
    for s in range(scenarios):
        for j in range(12):solver.changeRowBounds(j,-np.inf,float(b[s*nr+j]))
        solver.run()
        if solver.getModelStatus()!=highs.HighsModelStatus.kOptimal:return float('inf'),0.
        residual=max(residual,solver.getSolution().col_value[-2])
    if residual>=1e-7:return residual,0.
    solver.changeColCost(size-2,0.);solver.changeColCost(size-1,-1.)
    solver.changeColBounds(size-2,0.,residual+1e-9);solver.changeColBounds(size-1,0.,np.inf)
    margin=np.inf
    for s in range(scenarios):
        for j in range(12):solver.changeRowBounds(j,-np.inf,float(b[s*nr+j]))
        solver.run()
        if solver.getModelStatus()!=highs.HighsModelStatus.kOptimal:return residual,0.
        margin=min(margin,solver.getSolution().col_value[-1])
    return residual,-margin

class ContactRequirements:
    def __init__(self, request):
        payload = request['payload']
        self.shape = ObjectShape(payload['shape'], payload['size'], payload['mass'])
        source = np.asarray(request['source']); self.rotation = pin.Quaternion(source[3:]).matrix()
        displacement = np.asarray(request['destination'])[:3]-source[:3]
        acceleration = [np.zeros(3)]
        for delta, duration in [(np.array([0.,0.,.15]),4.), (displacement,6.)]:
            peak = 10/np.sqrt(3)*delta/duration**2
            acceleration.extend([peak, -peak])
        force = self.shape.mass*(np.asarray(acceleration)+[0.,0.,9.80665])
        self.wrenches = np.c_[force, np.zeros_like(force)]
        self.scale = np.r_[np.full(3,self.shape.mass*9.80665),
                           np.full(3,self.shape.mass*9.80665*np.linalg.norm(self.shape.size)/2)]
        directions = []
        for axis in self.rotation.T:
            directions.extend([axis, -axis])
        for tilt in [0., .45, -.45]:
            for angle in np.arange(8)*np.pi/4:
                directions.append(np.array([np.cos(angle)*np.cos(tilt),np.sin(angle)*np.cos(tilt),np.sin(tilt)]))
        sites = []
        for direction in directions:
            point, normal = self.shape.ray_surface(self.rotation.T@direction)
            if (self.rotation@point+source[:3])[2] < request['support_top']+.03:
                continue
            if any(np.linalg.norm(point-p)<1e-7 for p,_ in sites):
                continue
            sites.append((point,-normal))
        self.points = np.array([s[0] for s in sites]); self.normals = np.array([s[1] for s in sites])
        self.columns = []
        for point, normal in sites:
            r = self.rotation@point; n = self.rotation@normal
            t = np.cross(n,np.eye(3)[np.argmin(abs(n))]); t /= np.linalg.norm(t)
            b = np.cross(n,t)
            # Diamond inner approximation, same friction and 30 N L1 pad cap
            # as the downstream fixed-pose force screen.
            rays = np.array([n+2.25*t,n-2.25*t,n+2.25*b,n-2.25*b]).T
            self.columns.append(np.vstack([rays,np.cross(r,rays.T).T])/self.scale[:,None])
        self.cache = {}
        self.programs = {}
        # The task's numerical basis is immutable. Identify it once instead
        # of serializing every growing contact matrix for each memo lookup.
        basis=[np.asarray(self.columns),self.wrenches,self.scale]
        raw=b''.join(str(a.shape).encode()+a.tobytes() for a in basis)
        self.basis_digest=hashlib.sha256(raw).hexdigest()

    def cost(self, indices):
        key = tuple(sorted(indices))
        if key in self.cache:
            return self.cache[key]
        value=memoized(dict(operation='contact_role_margin_v3',basis=self.basis_digest,sites=key),
            lambda:self._cost(np.concatenate([self.columns[i] for i in key],axis=1),len(key)))
        self.cache[key]=value
        return value

    def _cost(self,matrix,contacts):
        k = matrix.shape[1]; ns = len(self.wrenches); width = k*ns+2
        # The last two variables are normalized wrench error and force margin.
        if contacts not in self.programs:
            rows=[]; bounds=[]
            for s,wrench in enumerate(self.wrenches/self.scale):
                for sign in [-1,1]:
                    row=np.zeros((6,width)); row[:,s*k:(s+1)*k]=0.; row[:,-2]=-1
                    rows.append(row); bounds.extend(sign*wrench)
                for contact in range(contacts):
                    # Unit-normal force >= .5 N and the existing conservative
                    # 30 N L1 cap. Erode each force halfspace by 30*rho*||a||.
                    constraints=[(-np.ones(4),-.5,1.),(3.25*np.ones(4),30.,3.25)]
                    for a in [-1,1]:
                        for b in [-1,1]:
                            constraints.append((2.25*(np.array([a,-a,b,-b])-1),0.,np.sqrt(2.25**2+2)))
                    for coefficients,limit,norm in constraints:
                        row=np.zeros((1,width));row[0,s*k+4*contact:s*k+4*(contact+1)]=coefficients
                        row[0,-1]=30.*norm;rows.append(row);bounds.append(limit)
            A=np.vstack(rows);b=np.asarray(bounds)
            self.programs[contacts]=(A,b)
        template,b=self.programs[contacts];A=template.copy()
        block=12+6*contacts
        for scenario in range(ns):
            start=scenario*block;columns=slice(scenario*k,(scenario+1)*k)
            A[start:start+6,columns]=-matrix
            A[start+6:start+12,columns]=matrix
        return _role_program(A,b,ns)

    def choose(self, maximum, rng):
        maximum=min(maximum,len(self.points))
        # Even perfectly aligned 30 N pad forces cannot provide more than
        # 30*k N. This is a necessary capacity bound, not a feasible grasp.
        minimum=max(1,int(np.ceil(np.max(np.linalg.norm(self.wrenches[:,:3],axis=1))/30.)))
        # Support functions bound each coarse site's vertical contribution.
        # This guides the proposal count; continuous surface motion means it
        # is NOT an impossibility bound for all structures of that count.
        capacity=np.sort([max(0.,float(np.max(A[2])))*self.scale[2]*30./3.25
                          for A in self.columns])[::-1]
        enough=np.flatnonzero(np.cumsum(capacity)>=float(np.max(self.wrenches[:,2])))
        preferred=max(minimum,int(enough[0]+1) if len(enough) else len(capacity))
        count=int(rng.integers(min(preferred,maximum),maximum+1))
        selected=[]; history=[]
        for _ in range(count):
            remaining=[i for i in range(len(self.points)) if i not in selected]
            results=np.array([self.cost(selected+[i]) for i in remaining])
            # Prefer a role that closes the task wrench deficit when available.
            if np.any(results[:,0]<1e-7):
                ids=np.flatnonzero(results[:,0]<1e-7)
                remaining=[remaining[i] for i in ids]; results=results[ids]
            costs=results[:,1] if np.all(results[:,0]<1e-7) else results[:,0]
            weights=np.exp(-(costs-np.min(costs))/.015)
            index=remaining[int(rng.choice(len(remaining),p=weights/weights.sum()))]
            selected.append(index); residual,cost=self.cost(selected)
            history.append(dict(site=index,residual=residual,force_margin=-cost))
        return self.points[selected],self.normals[selected],dict(
            wrenches=self.wrenches.tolist(),normalization=self.scale.tolist(),
            selected_sites=selected,role_trace=history,object_force_model_only=True,
            count_law='uniform_coarse_vertical_capacity_count_to_modules',requested_contacts=count,
            necessary_contact_count=minimum,
            preferred_contact_count=preferred,coarse_friction=2.25,
            normal_floor_N=.5,margin_scale_N=30.,
            margin_scope='distance of allocated force to force limits; not external wrench robustness')


@lru_cache(maxsize=8)
def _requirements(encoded):
    return ContactRequirements(json.loads(encoded))


def task_requirements(request):
    physical = {key:request[key] for key in ['payload','source','destination','support_top']}
    return _requirements(json.dumps(physical,sort_keys=True))
