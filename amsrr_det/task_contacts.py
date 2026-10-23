"""Object wrench requirements select contact roles before robot construction.

The small force-cone program ignores the robot. Its residual is a proposal
priority, not a claim that an assembled robot can realize those forces.
"""
from functools import lru_cache
import json
import numpy as np
import pinocchio as pin
from .linear_program import linprog
from .objects import ObjectShape
from .planning_memo import evaluate as memoized


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
            rays = np.array([n+4.5*t,n-4.5*t,n+4.5*b,n-4.5*b]).T
            self.columns.append(np.vstack([rays,np.cross(r,rays.T).T])/self.scale[:,None])
        self.cache = {}

    def cost(self, indices):
        key = tuple(sorted(indices))
        if key in self.cache:
            return self.cache[key]
        matrix = np.concatenate([self.columns[i] for i in key],axis=1)
        value=memoized(dict(operation='contact_role_margin_v2',matrix=matrix.tolist(),
            wrenches=self.wrenches.tolist(),scale=self.scale.tolist(),mass=self.shape.mass),
            lambda:self._cost(matrix,len(key)))
        self.cache[key]=value
        return value

    def _cost(self,matrix,contacts):
        k = matrix.shape[1]; ns = len(self.wrenches); width = k*ns+2
        # The last two variables are normalized wrench error and force margin.
        rows=[]; bounds=[]
        for s,wrench in enumerate(self.wrenches/self.scale):
            for sign in [-1,1]:
                row=np.zeros((6,width)); row[:,s*k:(s+1)*k]=sign*matrix; row[:,-2]=-1
                rows.append(row); bounds.extend(sign*wrench)
            for contact in range(contacts):
                # Unit-normal force >= .5 N and the existing conservative
                # 30 N L1 cap. Erode each force halfspace by 30*rho*||a||.
                constraints=[(-np.ones(4),-.5,1.),(5.5*np.ones(4),30.,5.5)]
                for a in [-1,1]:
                    for b in [-1,1]:
                        constraints.append((4.5*(np.array([a,-a,b,-b])-1),0.,np.sqrt(4.5**2+2)))
                for coefficients,limit,norm in constraints:
                    row=np.zeros((1,width));row[0,s*k+4*contact:s*k+4*(contact+1)]=coefficients
                    row[0,-1]=30.*norm;rows.append(row);bounds.append(limit)
        A=np.vstack(rows);b=np.asarray(bounds)
        objective=np.zeros(width);objective[-2]=1.
        limits=[(0,None)]*(width-1)+[(0,0)]
        result=linprog(objective,A_ub=A,b_ub=b,bounds=limits)
        if not result.success:return (float('inf'),0.)
        residual=float(result.x[-2])
        # Only compare force margins once the common task wrench is met.
        if residual>=1e-7:return residual,0.
        objective[-2]=0.;objective[-1]=-1.
        limits[-2]=(0,residual+1e-9);limits[-1]=(0,None)
        margin=linprog(objective,A_ub=A,b_ub=b,bounds=limits)
        return residual,float(-margin.x[-1]) if margin.success else 0.

    def choose(self, maximum, rng):
        maximum=min(maximum,len(self.points))
        count=int(rng.integers(1,maximum+1))
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
            count_law='uniform_1_to_modules',requested_contacts=count,
            normal_floor_N=.5,margin_scale_N=30.,
            margin_scope='distance of allocated force to force limits; not external wrench robustness')


@lru_cache(maxsize=8)
def _requirements(encoded):
    return ContactRequirements(json.loads(encoded))


def task_requirements(request):
    physical = {key:request[key] for key in ['payload','source','destination','support_top']}
    return _requirements(json.dumps(physical,sort_keys=True))
