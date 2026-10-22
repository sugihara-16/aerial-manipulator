"""Object-driven shared base/Dock references; geometry is not a sticking constraint."""
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import trimesh
from functools import lru_cache
from io import BytesIO
from scipy.optimize import lsq_linear,minimize,LinearConstraint,Bounds
from .linear_program import linprog
from .model import ROOT, transform
from .grasp_geometry import grasp_frame
from .equilibrium import point_jacobian
from .numerics import advance_world_orientation


@lru_cache(maxsize=4)
def _source_vertices(raw, file_type):
    """Cache only immutable authored geometry, keyed by its exact source bytes."""
    vertices = np.asarray(trimesh.load_mesh(BytesIO(raw), file_type=file_type).vertices).copy()
    vertices.setflags(write=False)
    return vertices


def surface_points(model,meta,q,surface):
    """Eight exposed material vertices, from the source mesh (not collision truth)."""
    module,port=surface
    link=('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]
    source=ROOT/'assets/robots/holon/holon.urdf'
    collision=ET.parse(source).find(f"link[@name='{link}']/collision")
    mesh=collision.find('geometry/mesh');path=source.parent/mesh.get('filename')
    source_vertices=_source_vertices(path.read_bytes(),path.suffix.lstrip('.'))
    frame=meta['frames'][grasp_frame(*surface)];body=meta['frames'][f'm{module}/{link}']
    if model.frames[frame].parentJoint!=model.frames[body].parentJoint:
        raise ValueError('grasp material frame must be fixed on its collision body')
    T=model.frames[frame].placement.inverse()*model.frames[body].placement*transform(collision.find('origin'))
    vertices=source_vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    vertices=vertices@T.rotation.T+T.translation
    front=vertices[vertices[:,0]>vertices[:,0].max()-.0001]
    angles=np.linspace(0,2*np.pi,8,endpoint=False)
    indices=np.unique([np.argmax(front[:,1]*np.cos(a)+front[:,2]*np.sin(a)) for a in angles])
    return frame,front[indices]


class GraspReferences:
    def __init__(self,model,meta,q,reference,surfaces,object_points,inward_normals,period=.01,samples=None,shape=None,active_only=False,posture_tracking=False):
        self.model,self.meta,self.dt=model,meta,period
        self.shape=shape;self.surfaces=surfaces;self.refresh_counter=0
        self.active_only=active_only;self.active_counts=[]
        self.posture_tracking=posture_tracking
        self.triangles=None
        if shape is not None:
            from .contact_geometry import material_triangles
            self.triangles=[material_triangles(port,meta.get('material_mesh_cache')) for _,port in surfaces]
        self.data=model.createData();self.reference=reference.copy()
        self.docks=[a for a in meta['actuators'] if a['role']=='dock']
        self.dock_actuators=np.array([i for i,a in enumerate(meta['actuators']) if a['role']=='dock'])
        self.qi=np.array([a['q'] for a in self.docks]);self.vi=np.array([a['v'] for a in self.docks])
        self.dock_lower=np.array([a['lower'] for a in self.docks]);self.dock_upper=np.array([a['upper'] for a in self.docks])
        self.dock_kp=np.array([a['kp'] for a in self.docks]);self.dock_kd=np.array([a['kd'] for a in self.docks])
        self.dock_effort=np.array([a['effort'] for a in self.docks])
        self.identity3=np.eye(3)
        self.samples=[surface_points(model,meta,q,s) for s in surfaces] if samples is None else list(samples)
        self.planes=np.asarray(object_points);self.normals=np.asarray(inward_normals)
        self.size=6+len(self.docks);self.last_velocity=np.zeros(self.size)
        self.scale=np.r_[np.full(3,.06),np.full(3,.15),np.full(len(self.docks),.2)]
        self.geometry_q=None;self.geometry_result=None
        _,_,_,compression=self.geometry(q)
        self.preload=np.maximum(compression,0.)
        self.initial=q.copy();self.last_diagnostics={}

    def geometry(self,q):
        # A phase guard and the reference update query the same observation.
        # Reusing it also keeps those extra reads from advancing patch refresh.
        if self.geometry_q is not None and np.array_equal(q,self.geometry_q):return self.geometry_result
        m,d=self.model,self.data;pin.computeJointJacobians(m,d,q);pin.updateFramePlacements(m,d)
        b=self.meta['bases'][0];R=d.oMi[b['joint']].rotation
        lift=np.zeros((m.nv,self.size));lift[b['v']:b['v']+3,:3]=R.T
        lift[b['v']+3:b['v']+6,3:6]=R.T;lift[self.vi,6+np.arange(len(self.docks))]=1
        obj=d.oMf[self.meta['object']['frame']];J=[];H=[];N=[];delta=[];depth_groups=[]
        self.active_counts=[]
        for j,((frame,points),plane,normal) in enumerate(zip(self.samples,self.planes,self.normals)):
            frame_jacobian=pin.getFrameJacobian(m,d,frame,pin.LOCAL_WORLD_ALIGNED)@lift
            n=obj.rotation@normal;origin=obj.act(plane);depths=[]
            if self.shape is not None:
                refresh=self.refresh_counter==0 or (self.refresh_counter-j*max(1,10//len(self.samples)))%10==0
                if refresh:
                    from .contact_geometry import close_material
                    points,nearest,gaps,outward=close_material(self.shape,obj.inverse()*d.oMf[frame],self.triangles[j])
                    self.samples[j]=(frame,points)
                else:
                    world=points@d.oMf[frame].rotation.T+d.oMf[frame].translation
                    local=(world-obj.translation)@obj.rotation
                    nearest,gaps,outward=self.shape.project_surface(local)
                centers=nearest@obj.rotation.T+obj.translation
                normals=-outward@obj.rotation.T
            normal_only=False
            if self.shape is not None and self.active_only:
                active=np.flatnonzero(gaps<=1e-7);self.active_counts.append(len(active))
                normal_only=len(active)==0
                selected=np.array([np.argmin(gaps)]) if normal_only else active
                points=points[selected];centers=centers[selected];normals=normals[selected];gaps=gaps[selected]
            first_point=len(J)
            for i,point in enumerate(points):
                world=d.oMf[frame].act(point)
                projection=np.outer(normals[i],normals[i]) if normal_only else self.identity3
                J.append(projection@(frame_jacobian[:3]-pin.skew(d.oMf[frame].rotation@point)@frame_jacobian[3:]))
                H.append(projection@np.column_stack([self.identity3,-pin.skew((centers[i] if self.shape is not None else world)-obj.translation)]))
                column=np.zeros((3,len(self.samples)));column[:,j]=normals[i] if self.shape is not None else n;N.append(column)
                depths.append(-gaps[i] if self.shape is not None else n@(world-origin))
            delta.append(np.max(depths) if self.shape is not None else np.mean(depths))
            depth_groups.append((first_point,len(depths),int(np.argmax(depths))))
        self.refresh_counter+=1
        self.geometry_q=q.copy();self.geometry_result=(np.vstack(J),np.vstack(H),np.vstack(N),np.array(delta))
        # Derivative of the SAME compression used above. On a real surface,
        # track the deepest material point; do not constrain the whole patch.
        self.compression_map=np.zeros((len(delta),3*len(J)))
        for pad,(first,count,deepest) in enumerate(depth_groups):
            rows=slice(3*(first+deepest),3*(first+deepest+1)) if self.shape is not None else slice(3*first,3*(first+count))
            self.compression_map[pad,rows]=self.geometry_result[2][rows,pad]/(1 if self.shape is not None else count)
        return self.geometry_result

    def relative_material_speed(self,q,v):
        """Largest observed pad/object relative speed at selected material points."""
        J,H,_,_=self.geometry(q);base=self.meta['bases'][0];obj=self.meta['object']
        R=pin.Quaternion(q[base['q']+3:base['q']+7]).matrix()
        Ro=pin.Quaternion(q[obj['q']+3:obj['q']+7]).matrix()
        bv,ov=base['v'],obj['v']
        robot=np.r_[R@v[bv:bv+3],R@v[bv+3:bv+6],v[self.vi]]
        payload=np.r_[Ro@v[ov:ov+3],Ro@v[ov+3:ov+6]]
        return float(np.linalg.norm((J@robot-H@payload).reshape(-1,3),axis=1).max())

    def update(self,q,v,object_target, *, desired_twist=None, preload=None, normal_speed=.001,recover_compression_only=False,orientation_feedback=1.,object_feedback=1.,releasing=False,drive_guard=None,allow_guard_braking=False,previous_joint_bias=None,preserve_preload=False):
        J,H,N,delta=self.geometry(q);obj=self.meta['object'];oq,ov=obj['q'],obj['v']
        R=pin.Quaternion(q[oq+3:oq+7]).matrix();Rd=pin.Quaternion(object_target[3:]).matrix()
        error=np.r_[object_target[:3]-q[oq:oq+3],pin.log3(Rd@R.T)]
        velocity=np.r_[R@v[ov:ov+3],R@v[ov+3:ov+6]]
        vd=np.zeros(6) if desired_twist is None else desired_twist
        vc=np.clip(vd+np.r_[np.ones(3),np.full(3,.8)]*error-.3*(velocity-vd),
                   [-.06]*3+[-.15]*3,[.06]*3+[.15]*3)
        vc[3:]*=orientation_feedback
        vc*=object_feedback
        inverse=np.linalg.pinv(H,rcond=1e-9);P=inverse@J
        depth_target=self.preload if preload is None else np.asarray(preload)
        # Recover missing compression without increasing the fixed target.
        # This bounds the request before projection, not each pad's motion.
        closing=np.clip(1.5*(depth_target-delta),0. if recover_compression_only else -normal_speed,normal_speed)
        # Fit common motion and relative closure with position/speed bounds in
        # the solve. A single joint at its bound must not freeze the whole base.
        common=H@P;internal=J-common
        squeeze=N@closing;internal_goal=squeeze-H@(inverse@squeeze)
        # During release each pad must be allowed to retreat. Suppressing the
        # rigid component of that motion can make side pads move down when an
        # upper pad opens, even with object feedback already switched off.
        common_goal=H@vc+(H@(inverse@squeeze) if releasing else 0.)
        A=np.vstack([4*common,internal,np.diag(.0001/self.scale)])*self.scale
        preferred=np.zeros(self.size)
        if self.posture_tracking and not releasing:
            # Moving the whole grasp rigidly already realizes object motion.
            # Use the remaining freedom to restore the acquired Dock posture,
            # rather than accumulating a new shape through minimum-speed IK.
            preferred[:3]=vc[:3]+np.cross(vc[3:],self.reference[:3]-q[oq:oq+3])
            preferred[3:6]=vc[3:]
            # Contact points alone leave a rotation about their common axis
            # weakly constrained. Retain the acquired base/object pose as well
            # as the Dock posture; the target moves with the observed object.
            acquired_object=pin.SE3(pin.Quaternion(self.initial[oq+3:oq+7]).matrix(),self.initial[oq:oq+3])
            acquired_base=pin.SE3(pin.Quaternion(self.initial[3:7]).matrix(),self.initial[:3])
            held_base=pin.SE3(R,q[oq:oq+3])*acquired_object.inverse()*acquired_base
            preferred[:3]+=held_base.translation-self.reference[:3]
            preferred[3:6]+=pin.log3(held_base.rotation@pin.Quaternion(self.reference[3:7]).matrix().T)
            preferred[6:]=self.initial[self.qi]-self.reference[self.qi]
        rhs=np.r_[4*common_goal,internal_goal,.0001*preferred/self.scale]
        limit=np.r_[np.full(3,.1),np.full(3,.3),np.full(len(self.docks),.5)]*self.dt
        lower=np.maximum(-self.scale,self.last_velocity-limit)
        upper=np.minimum(self.scale,self.last_velocity+limit)
        hard_lower=-self.scale.copy();hard_upper=self.scale.copy()
        ref=self.reference[self.qi]
        low=(self.dock_lower+.025-ref)/self.dt;high=(self.dock_upper-.025-ref)/self.dt
        tracking_error=ref-q[self.qi]
        high=np.where(tracking_error>.02,np.minimum(high,0.),high)
        low=np.where(tracking_error<-.02,np.maximum(low,0.),low)
        # Position/tracking bounds override smooth acceleration on recovery.
        lower[6:]=np.clip(lower[6:],low,high);upper[6:]=np.clip(upper[6:],low,high)
        hard_lower[6:]=np.clip(hard_lower[6:],low,high);hard_upper[6:]=np.clip(hard_upper[6:],low,high)
        command_bounds=None;command_step_limit=None;relaxed_docks=[]
        if drive_guard is not None:
            bias,command=drive_guard
            kp=self.dock_kp;kd=self.dock_kd
            effort=self.dock_effort
            raw=kp*(np.asarray(command)-q[self.qi])-kd*v[self.vi]
            constant=kp*(self.reference[self.qi]-q[self.qi])+np.asarray(bias)-kd*v[self.vi]
            command_bounds=(np.minimum(-effort,raw),np.maximum(effort,raw))
            lo_drive=(command_bounds[0]-constant)/(kp*self.dt+kd)
            hi_drive=(command_bounds[1]-constant)/(kp*self.dt+kd)
            lower[6:]=np.maximum(lower[6:],lo_drive);upper[6:]=np.minimum(upper[6:],hi_drive)
            hard_lower[6:]=np.maximum(hard_lower[6:],lo_drive);hard_upper[6:]=np.minimum(hard_upper[6:],hi_drive)
            if allow_guard_braking:
                if previous_joint_bias is None:raise ValueError('previous applied joint bias is required for command-rate bounds')
                # Preserve the command increment already allowed by the old
                # speed/acceleration limits, including the known bias change.
                command_step_limit=self.dt*self.scale[6:]+kd/kp*limit[6:]+abs(np.asarray(bias)-previous_joint_bias)/kp
                lo_step=(kp*(np.asarray(command)-command_step_limit-self.reference[self.qi])-bias)/(kp*self.dt+kd)
                hi_step=(kp*(np.asarray(command)+command_step_limit-self.reference[self.qi])-bias)/(kp*self.dt+kd)
                lower[6:]=np.maximum(lower[6:],lo_step);upper[6:]=np.minimum(upper[6:],hi_step)
                hard_lower[6:]=np.maximum(hard_lower[6:],lo_step);hard_upper[6:]=np.minimum(hard_upper[6:],hi_step)
                conflict=(lower[6:]>upper[6:]+1e-12)&(hard_lower[6:]<=hard_upper[6:])
                indices=6+np.flatnonzero(conflict)
                lower[indices]=hard_lower[indices];upper[indices]=hard_upper[indices]
                relaxed_docks=[a['name'] for a,flag in zip(self.docks,conflict) if flag]
            if np.any(lower>upper+1e-12):
                self.last_diagnostics=dict(reference_command_feasible=False,conflicting_docks=[a['name'] for a,lo,hi in zip(self.docks,lower[6:],upper[6:]) if lo>hi+1e-12],reference_velocity_lower=lower.tolist(),reference_velocity_upper=upper.tolist(),hard_velocity_lower=hard_lower.tolist(),hard_velocity_upper=hard_upper.tolist(),drive_velocity_lower=lo_drive.tolist(),drive_velocity_upper=hi_drive.tolist())
                tv=np.zeros(self.model.nv);tv[:6]=self.last_velocity[:6];tv[self.vi]=self.last_velocity[6:]
                return self.reference.copy(),tv
        lo,hi=lower/self.scale,upper/self.scale;free=hi-lo>1e-12
        z=(lo+hi)/2
        if np.any(free):
            z[free]=lsq_linear(A[:,free],rhs-A[:,~free]@z[~free],bounds=(lo[free],hi[free]),
                               method='bvls',tol=1e-8,max_iter=40).x
        preload_check={};preload_condition=None
        if preserve_preload and not releasing:
            weak=delta<depth_target
            if np.any(weak):
                C=self.compression_map[weak]
                rate=np.maximum(0.,C@internal_goal)
                normal_map=(C@J)*self.scale;required=C@H@velocity+rate
                iterations=0
                optimal=True
                if np.min(normal_map@z-required)<-1e-9:
                    # Whiten the same quadratic objective; its small posture
                    # tie-breaker otherwise gives a badly conditioned Hessian.
                    # This changes coordinates, not priorities or bounds.
                    transform=np.linalg.solve(np.linalg.cholesky(A.T@A).T,np.eye(self.size))
                    matrix=np.vstack([np.eye(self.size),1000*normal_map])
                    lower_qp=np.r_[lo-z,1000*(required-normal_map@z)]
                    upper_qp=np.r_[hi-z,np.full(len(required),np.inf)]
                    gradient=transform.T@(A.T@(A@z-rhs))
                    solved=minimize(lambda y:.5*y@y+gradient@y,np.zeros(self.size),jac=lambda y:y+gradient,
                        constraints=[LinearConstraint(matrix@transform,lower_qp,upper_qp)],
                        method='SLSQP',options=dict(ftol=1e-10,maxiter=60))
                    iterations=int(solved.nit);optimal=bool(solved.success)
                    unconstrained_candidate=z+transform@solved.x
                    # Remove numerical box-bound residuals before checking the
                    # closing condition. Projection never expands a bound.
                    candidate=np.clip(unconstrained_candidate,lo,hi)
                    # A feasible iterate need not have reached the optimum.
                    # Independently verify every unchanged physical bound.
                    valid=(np.isfinite(candidate).all()
                           and np.min(normal_map@candidate-required)>=-1e-9)
                    if not valid:
                        self.preload_failure_problem=dict(A=A,rhs=rhs,lower=lo,upper=hi,
                            normal_map=normal_map,required=required,initial=z,candidate=candidate,
                            unprojected_candidate=unconstrained_candidate)
                        self.last_diagnostics=dict(preload_reference_feasible=False,reason=str(solved.message),
                            minimum_closing_margin_m_s=float(np.min(normal_map@candidate-required)))
                        tv=np.zeros(self.model.nv);tv[:6]=self.last_velocity[:6];tv[self.vi]=self.last_velocity[6:]
                        return self.reference.copy(),tv
                    z=candidate
                preload_condition=(normal_map,required)
                preload_check=dict(preload_reference_feasible=True,preload_guard_iterations=iterations,
                    preload_guard_optimality_converged=optimal,
                    guarded_pads=np.flatnonzero(weak).tolist(),required_closing_rate_m_s=rate.tolist())
        release_feasible=True;outward=None;release_smoothing_relaxed=False;release_solver={}
        if releasing:
            outward=-N.sum(axis=1).reshape(-1,3)
            normal_map=np.einsum('ij,ijk->ik',outward,J.reshape(-1,3,self.size))*self.scale
            object_normal=np.einsum('ij,ij->i',outward,(H@velocity).reshape(-1,3))
            if np.min(normal_map@z-object_normal)<-1e-9:
                # A bound can defeat the outward least-squares target. Enforce
                # non-closing motion relative to the observed moving object.
                C=normal_map*1000;lower_normal=object_normal*1000
                # Reference acceleration smoothing is a design preference;
                # reference speed/position and actuator limits remain fixed.
                # A feasibility solve separates incompatible bounds from an
                # optimizer that merely failed to find a feasible starting point.
                for candidate_lo,candidate_hi in [(lo,hi),(hard_lower/self.scale,hard_upper/self.scale)]:
                    feasible=linprog(np.zeros(self.size),A_ub=-C,b_ub=-lower_normal,
                        bounds=list(zip(candidate_lo,candidate_hi)),method='highs')
                    if feasible.success:break
                    release_smoothing_relaxed=True
                release_feasible=bool(feasible.success)
                if release_feasible:
                    result=minimize(lambda x:.5*np.sum((A@x-rhs)**2),np.clip(z,candidate_lo,candidate_hi),
                        jac=lambda x:A.T@(A@x-rhs),bounds=Bounds(candidate_lo,candidate_hi),
                        constraints=[LinearConstraint(C,lower_normal,np.inf)],method='SLSQP',
                        options=dict(ftol=1e-12,maxiter=60))
                    use_qp=np.min(normal_map@result.x-object_normal)>=-1e-8
                    release_solver=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                        used_feasible_vertex=not bool(use_qp),objective=float(result.fun),
                        target_speed_m_s=float(np.linalg.norm(common_goal.reshape(-1,3),axis=1).max()))
                    z=result.x if use_qp else feasible.x
        nu=z*self.scale
        if not release_feasible:nu[:]=0.
        target=self.reference.copy();target[:3]+=nu[:3]*self.dt
        target[3:7]=advance_world_orientation(target[3:7],nu[3:6]*self.dt)
        target[self.qi]+=nu[6:]*self.dt
        base_error=np.linalg.norm(target[:3]-q[:3]);previous_base=np.linalg.norm(self.reference[:3]-q[:3])
        stalled=base_error>.05 and base_error>previous_base+1e-10
        if stalled:
            target=self.reference.copy();nu[:]=0.
        self.reference=target;self.last_velocity=nu.copy()
        tv=np.zeros(self.model.nv);tv[:6]=nu[:6];tv[self.vi]=nu[6:]
        self.last_diagnostics=dict(object_error=error.tolist(),compression=delta.tolist(),
                                   preload=depth_target.tolist(),stalled=bool(stalled),
                                   active_material_points=self.active_counts.copy(),
                                   compatibility_residual=float(np.linalg.norm(J@nu-H@(P@nu)-N@closing)))
        if preload_condition is not None:
            normal_map,required=preload_condition
            margin=float(np.min(normal_map@(nu/self.scale)-required))
            preload_check.update(preload_reference_feasible=margin>=-1e-9,minimum_closing_margin_m_s=margin)
            self.last_diagnostics.update(preload_check)
        if command_bounds is not None:
            predicted=constant+(kp*self.dt+kd)*nu[6:]
            self.last_diagnostics.update(reference_command_feasible=bool(np.all(predicted>=command_bounds[0]-1e-9) and np.all(predicted<=command_bounds[1]+1e-9)),
                predicted_drive_effort_nm=predicted.tolist(),
                previous_drive_effort_nm=raw.tolist())
            if command_step_limit is not None:
                issued=target[self.qi]+(np.asarray(bias)+kd*nu[6:])/kp
                self.last_diagnostics.update(reference_acceleration_relaxed_docks=relaxed_docks,
                    previous_commands=np.asarray(command).tolist(),command_step_limits=command_step_limit.tolist(),
                    drive_request_lower=command_bounds[0].tolist(),drive_request_upper=command_bounds[1].tolist())
                self.last_diagnostics['reference_command_feasible'] &= bool(np.all(abs(issued-command)<=command_step_limit+1e-9))
        if releasing:
            release_feasible=release_feasible and bool(np.min(np.einsum('ij,ij->i',outward,(J@nu-H@velocity).reshape(-1,3)))>=-1e-8)
            self.last_diagnostics.update(release_direction_feasible=release_feasible,
                release_smoothing_relaxed=release_smoothing_relaxed,
                release_solver=release_solver,
                minimum_reference_outward_speed_m_s=float(np.min(np.einsum('ij,ij->i',outward,(J@nu-H@velocity).reshape(-1,3)))))
        return target.copy(),tv
