"""Known convex object geometry and mass properties for grasp/transport inputs."""
from dataclasses import dataclass
import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial import ConvexHull
from .native_grasp_geometry import ellipsoid_projection,triangle_projection


@dataclass
class ObjectShape:
    family: str
    size: np.ndarray
    mass: float
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    inertia: np.ndarray | None = None

    def __post_init__(self):
        self.size=np.asarray(self.size,dtype=float)
        if self.size.shape!=(3,) or np.any(self.size<=0) or self.mass<=0:raise ValueError('invalid object')
        a,b,c=self.size
        if self.family in ['box','elongated_box']:
            self.inertia=self.mass/12*np.diag([b*b+c*c,a*a+c*c,a*a+b*b])
        elif self.family=='ellipsoid':
            self.inertia=self.mass/20*np.diag([b*b+c*c,a*a+c*c,a*a+b*b])
        elif self.family=='cylinder':
            if abs(a-b)>1e-10:raise ValueError('cylinder needs equal horizontal diameters')
            self.inertia=np.diag([self.mass*(3*(a/2)**2+c*c)/12]*2+[self.mass*(a/2)**2/2])
        elif self.family in ['triangular_prism','tapered_box']:
            if self.family=='triangular_prism':
                self.vertices=np.array([[x,y,z] for z in [-c/2,c/2] for x,y in [(-a/2,-b/2),(a/2,-b/2),(0,b/2)]])
            else:
                self.vertices=np.array([[x*a*s/2,y*b*s/2,z*c/2] for z,s in [(-1,1),(1,.6)] for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]])
            hull=trimesh.convex.convex_hull(self.vertices)
            self.vertices=np.asarray(hull.vertices)-hull.center_mass;self.faces=np.asarray(hull.faces)
            self.inertia=np.asarray(hull.moment_inertia)*self.mass/hull.mass
            self.equations=np.unique(np.round(ConvexHull(self.vertices).equations,14),axis=0)
            self.triangles=self.vertices[self.faces]
        else:raise ValueError('unsupported shape family')

    def payload(self):
        p=dict(shape=self.family,size=self.size.tolist(),mass=self.mass,inertia=self.inertia.tolist())
        if self.vertices is not None:p.update(vertices=self.vertices.tolist(),faces=self.faces.tolist())
        return p

    def lowest_z(self,pose):
        R=pin.Quaternion(np.asarray(pose)[3:]).matrix();a=self.size/2
        if self.family in ['box','elongated_box']:extent=abs(R[2])@a
        elif self.family=='ellipsoid':extent=np.linalg.norm(R[2]*a)
        elif self.family=='cylinder':extent=a[0]*np.linalg.norm(R[2,:2])+a[2]*abs(R[2,2])
        else:return float(pose[2]+(self.vertices@R[2]).min())
        return float(pose[2]-extent)

    def ray_surface(self,direction):
        """Ray from the COM to a real surface, returning its point/outward normal."""
        u=np.asarray(direction,dtype=float);u=u/np.linalg.norm(u);a=self.size/2
        if self.family in ['box','elongated_box']:
            dist=np.divide(a,abs(u),out=np.full(3,np.inf),where=abs(u)>1e-12)
            i=np.argmin(dist);p=dist[i]*u;n=np.zeros(3);n[i]=np.sign(u[i])
        elif self.family=='ellipsoid':
            p=u/np.linalg.norm(u/a);n=p/(a*a);n/=np.linalg.norm(n)
        elif self.family=='cylinder':
            side=a[0]/max(np.linalg.norm(u[:2]),1e-15);cap=a[2]/max(abs(u[2]),1e-15)
            p=min(side,cap)*u;n=np.r_[p[:2]/a[0],0.] if side<cap else np.array([0.,0.,np.sign(u[2])])
        else:
            equations=self.equations
            denom=equations[:,:3]@u
            ds=np.divide(-equations[:,3],denom,out=np.full(len(denom),np.inf),where=denom>1e-10)
            i=np.argmin(ds);p=ds[i]*u;n=equations[i,:3]
        return p,n

    def opposed_sites(self,pose,axis_world,tilt=0.):
        """Opposing sides, optionally inclined together above/below the COM."""
        if not np.isfinite(tilt) or abs(tilt)>=np.pi/2:raise ValueError("invalid contact inclination")
        axis=np.asarray(axis_world,dtype=float);axis=axis/np.linalg.norm(axis)
        if tilt and abs(axis[2])>1e-8:raise ValueError("inclination requires a horizontal grasp axis")
        R=pin.Quaternion(np.asarray(pose)[3:]).matrix()
        points=[];normals=[]
        for sign in [1.,-1.]:
            direction=sign*np.cos(tilt)*axis+np.array([0.,0.,np.sin(tilt)])
            point,normal=self.ray_surface(R.T@direction)
            points.append(point);normals.append(-normal)
        return np.array(points),np.array(normals)

    def project_surface(self,points):
        """Closest points, signed Euclidean gaps and outward normals, in COM axes.

        Negative gap means inside. This uses known geometry, never a simulator
        contact manifold. At sharp features the separating normal is returned.
        """
        x=np.atleast_2d(np.asarray(points,dtype=float));a=self.size/2
        if self.family in ['box','elongated_box']:
            p=np.clip(x,-a,a);inside=np.all(abs(x)<=a,axis=1)
            row=np.flatnonzero(inside);axis=np.argmin(a-abs(x[row]),axis=1)
            p[row,axis]=np.where(x[row,axis]>=0,1.,-1.)*a[axis]
        elif self.family=='cylinder':
            radius=np.linalg.norm(x[:,:2],axis=1);inside=(radius<=a[0])&(abs(x[:,2])<=a[2])
            direction=np.divide(x[:,:2],radius[:,None],out=np.tile([1.,0.],(len(x),1)),where=radius[:,None]>1e-14)
            p=x.copy();p[:,:2]=direction*np.minimum(radius,a[0])[:,None];p[:,2]=np.clip(x[:,2],-a[2],a[2])
            side=inside&((a[0]-radius)<a[2]-abs(x[:,2]));cap=inside&~side
            p[side,:2]=direction[side]*a[0];p[cap,2]=np.where(x[cap,2]>=0,1.,-1.)*a[2]
        elif self.family=='ellipsoid' and (native_points:=ellipsoid_projection(x,a)) is not None:
            inside=np.sum((x/a)**2,axis=1)<1;p=native_points
        elif self.family=='ellipsoid':
            inside=np.sum((x/a)**2,axis=1)<1
            # Lagrange multiplier of the nearest-point constraint. Handle the
            # interior medial set separately, where the shortest-axis component
            # can be nonzero despite an exactly zero query component.
            aa=a*a;amin=aa.min();low=np.full(len(x),-amin+1e-15)
            high=np.maximum(np.linalg.norm(x*a,axis=1),amin)
            value=lambda lam:np.sum((x*a/(lam[:,None]+aa))**2,axis=1)
            medial=value(low)<1
            lam=np.zeros(len(x))
            for _ in range(12):
                denom=lam[:,None]+aa
                terms=(x*a/denom)**2;residual=terms.sum(axis=1)-1
                low=np.where(residual>0,lam,low);high=np.where(residual<=0,lam,high)
                derivative=-2*np.sum(terms/denom,axis=1)
                trial=lam-np.divide(residual,derivative,out=np.zeros_like(residual),where=abs(derivative)>1e-20)
                converged=abs(residual)<1e-12
                lam=np.where(converged,lam,np.where((trial>low)&(trial<high),trial,(low+high)/2))
                # Later iterations leave converged multipliers unchanged.
                # Medial-set points are resolved independently below.
                if np.all(converged|medial):break
            unresolved=(abs(value(lam)-1)>1e-10)&~medial
            if np.any(unresolved):
                for _ in range(40):
                    mid=(low+high)/2;outside=value(mid)>1
                    low=np.where(outside,mid,low);high=np.where(outside,high,mid)
                lam=np.where(unresolved,(low+high)/2,lam)
            p=aa*x/(lam[:,None]+aa)
            if np.any(medial):
                ids=np.flatnonzero(medial);short=int(np.argmin(aa));other=aa>amin+1e-14
                p[ids]=0.;p[np.ix_(ids,other)]=aa[other]*x[np.ix_(ids,other)]/(aa[other]-amin)
                p[ids,short]=a[short]*np.sqrt(np.maximum(0,1-np.sum((p[ids]/a)**2,axis=1)))
        else:
            equations=self.equations;values=x@equations[:,:3].T+equations[:,3]
            closest=np.argmax(values,axis=1);signed=values[np.arange(len(x)),closest]
            p=x-signed[:,None]*equations[closest,:3];inside=signed<0
            # A projection inside a face is already the exact closest point.
            # Only edge/vertex cases need the more expensive triangle query.
            unresolved=np.max(p@equations[:,:3].T+equations[:,3],axis=1)>1e-12
            if np.any(unresolved):
                triangles=self.triangles;xx=x[unresolved]
                projected=triangle_projection(xx,triangles)
                if projected is not None:p[unresolved]=projected
                else:
                    tiled=np.broadcast_to(triangles,(len(xx),*triangles.shape)).reshape(-1,3,3)
                    candidates=trimesh.triangles.closest_point(tiled,np.repeat(xx,len(triangles),axis=0)).reshape(len(xx),-1,3)
                    ids=np.argmin(np.linalg.norm(candidates-xx[:,None,:],axis=2),axis=1)
                    p[unresolved]=candidates[np.arange(len(xx)),ids]
        distance=np.linalg.norm(x-p,axis=1);gap=np.where(inside,-distance,distance)
        normals=np.divide(x-p,gap[:,None],out=np.zeros_like(x),where=abs(gap[:,None])>1e-12)
        boundary=distance<=1e-12
        if np.any(boundary):
            for i in np.flatnonzero(boundary):normals[i]=self.ray_surface(p[i])[1]
        return p,gap,normals
