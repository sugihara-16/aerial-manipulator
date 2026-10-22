"""Experimental local derivative of observed-only MuJoCo contact response.

The derivative freezes contact locations, normals and active friction/drive
branches. It is a candidate-generation approximation, never a force guarantee.
"""
import numpy as np
import mujoco
import pinocchio as pin
from scipy.linalg import cho_factor, cho_solve


def contact_hessian(model, data, jar):
    """Derivative of the force law at this independent model's contact rows."""
    m,d=model,data
    mujoco.mj_constraintUpdate(m,d,jar,None,1)
    C=np.diag(np.where(d.efc_state==mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC,d.efc_D,0.))
    ids=np.unique(d.efc_id[d.efc_state==mujoco.mjtConstraintState.mjCNSTRSTATE_CONE])
    if len(ids):
        address=np.asarray(d.contact.efc_address)[ids]
        dimensions=np.asarray(d.contact.dim)[ids]
        for size in np.unique(dimensions):
            use=dimensions==size;rows=address[use,None]+np.arange(size)
            C[rows[:,:,None],rows[:,None,:]]=np.asarray(d.contact.H)[ids[use],:size*size].reshape(-1,size,size)
    return C


def local_input_tangent(model, data):
    m, d = model, data
    if m.na or np.any(m.actuator_dyntype) or np.any(m.actuator_gaintype):
        raise ValueError('local tangent requires stateless fixed-gain drives')
    if np.any(m.jnt_actfrclimited) or m.ntendon or m.nflex:
        raise ValueError('unsupported effective metric or extra force limits')
    if not d.efm_active or mujoco.mj_isSparse(m):
        raise ValueError('diagnostic requires dense discrete effective metric')
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    M += np.diag(d.efm_sdiag)
    J = d.efc_J.reshape(d.nefc, m.nv).copy()
    jar = J @ d.qacc - d.efc_aref
    C = contact_hessian(m,d,jar)
    moment = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(moment, d.actuator_moment,
        d.moment_rownnz, d.moment_rowadr, d.moment_colind)
    raw = m.actuator_gainprm[:, 0]*d.ctrl + m.actuator_biasprm[:, 1]*d.actuator_length + m.actuator_biasprm[:, 2]*d.actuator_velocity
    free = (~m.actuator_forcelimited.astype(bool)) | ((raw > m.actuator_forcerange[:, 0]) & (raw < m.actuator_forcerange[:, 1]))
    gain = m.actuator_gainprm[:, 0]*free
    H = M + J.T @ C @ J
    factor = cho_factor(H, lower=True)
    acceleration = cho_solve(factor, moment.T*gain)
    forces = -C @ J @ acceleration
    balance = M @ d.qacc - (d.qfrc_smooth + d.efm_c + d.efm_ca + J.T @ d.efc_force)
    return dict(acceleration_wrt_control=acceleration, force_wrt_control=forces,
                mass=M, jacobian=J, contact_derivative=C, moment=moment,
                factor=factor, free_drives=free,
                dynamic_balance_max=float(abs(balance).max(initial=0.)))


def frozen_response(plant, q, v, first, tangent):
    """Affine discrete model, with rigid/drive and distance/velocity derivatives.

    Contact Jacobians, regularization/impedance and force-law branches are
    fixed at the first step. No geometry or force is fetched from a real plant.
    """
    m, d, model = plant.mj, first, plant.pin
    h, n = m.opt.timestep, m.nv
    S = np.zeros((n, n))  # MuJoCo velocities to Pinocchio local velocities.
    Vq = np.zeros((n,n));Aq = np.zeros((n,n));Av = np.zeros((n,n))
    correction = np.zeros(n);bases=[]
    for jid,mjid in plant.joints:
        joint=model.joints[jid];j,k=joint.idx_v,m.jnt_dofadr[mjid]
        if joint.nq==7:
            R=pin.Quaternion(q[joint.idx_q+3:joint.idx_q+7]).matrix()
            S[j:j+3,k:k+3]=R.T;S[j+3:j+6,k+3:k+6]=np.eye(3)
            linear,angular=v[j:j+3],v[j+3:j+6]
            correction[j:j+3]=-np.cross(angular,linear)
            Vq[j:j+3,k+3:k+6]=pin.skew(linear)
            Aq[j:j+3,k+3:k+6]=pin.skew(R.T@d.qacc[k:k+3])-pin.skew(angular)@pin.skew(linear)
            Av[j:j+3,k:k+3]=-pin.skew(angular)@R.T
            Av[j:j+3,k+3:k+6]=pin.skew(linear)
            bases.append((j,k,R))
        else:S[j,k]=1.
    acceleration=S@d.qacc+correction
    external=[pin.Force.Zero() for _ in range(model.njoints)]
    for i,rotor in enumerate(plant.meta['rotors']):
        frame=model.frames[rotor['frame']]
        force=d.ctrl[m.actuator(f'rotor{i}').id]
        external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,force]),np.zeros(3)))
    pd=model.createData()
    rq,rv,mass=map(np.array,pin.computeRNEADerivatives(model,pd,q,v,acceleration,external))
    force=pin.rnea(model,pd,q,v,acceleration,external).copy()
    Rq=S.T@(rq@S+rv@Vq+mass@Aq)
    Rv=S.T@(rv@S+mass@Av)
    for j,k,R in bases:
        Rq[k:k+3,k+3:k+6]-=R@pin.skew(force[j:j+3])
    moment=tangent['moment'];free=tangent['free_drives']
    kp=-m.actuator_biasprm[:,1]*free;kd=-m.actuator_biasprm[:,2]*free
    Rq+=moment.T@(kp[:,None]*moment)
    Rv+=moment.T@((kd+h*kp)[:,None]*moment)
    rigid_q,rigid_v=Rq.copy(),Rv.copy()
    J,C=tangent['jacobian'],tangent['contact_derivative']
    K=d.efc_KBIP[:,0]*d.efc_KBIP[:,2];D=d.efc_KBIP[:,1]
    denominator=1+h*D+h*h*K
    position=K/denominator;velocity=(D+h*K)/denominator
    # K is zero for tangential rows; only normal gap has positional stiffness.
    Rq+=J.T@C@(position[:,None]*J)
    Rv+=J.T@C@(velocity[:,None]*J)
    aq=-cho_solve(tangent['factor'],Rq)
    av=-cho_solve(tangent['factor'],Rv)
    au=tangent['acceleration_wrt_control']
    Q=np.eye(n);V=h*np.eye(n)
    for _,k,_ in bases:
        a=np.empty(9);b=np.empty(9);c=np.empty(3)
        mujoco.mjd_quatIntegrate(d.qvel[k+3:k+6],h,a,b,c)
        Q[k+3:k+6,k+3:k+6]=a.reshape(3,3)
        # MuJoCo returns the Jacobian w.r.t. the scaled rotation increment;
        # its Dvel is identity at zero even when scale=h. Apply h explicitly.
        V[k+3:k+6,k+3:k+6]=h*b.reshape(3,3)
    A=np.block([[Q+V@(h*aq),V@(np.eye(n)+h*av)],
                [h*aq,np.eye(n)+h*av]])
    B=np.vstack([V@(h*au),h*au])
    q0=plant.configuration(q);v0=S.T@v
    shift=np.empty(n);mujoco.mj_differentiatePos(m,shift,1.,q0,d.qpos)
    offset=np.r_[shift,d.qvel-v0]
    Lx=-C@np.hstack([J@aq+position[:,None]*J,J@av+velocity[:,None]*J])
    Lu=-C@J@au
    return dict(A=A,B=B,offset=offset,Lx=Lx,Lu=Lu,force=d.efc_force.copy(),
                q0=q0,v0=v0,rigid_coordinate_map=S,
                rigid_residual_wrt_position=rigid_q,rigid_residual_wrt_velocity=rigid_v,
                position_coefficient=position,velocity_coefficient=velocity,
                limitations='Contact points/normals/J, impedance/regularization and branches fixed; no relinearization.')


def propagate(model, response, delta, steps=40):
    x=np.zeros(2*model.nv);forces=[];states=[]
    for _ in range(steps):
        forces.append(response['force']+response['Lx']@x+response['Lu']@delta)
        x=response['A']@x+response['B']@delta+response['offset']
        states.append(x.copy())
    q=response['q0'].copy();mujoco.mj_integratePos(model,q,x[:model.nv],1.)
    return q,response['v0']+x[model.nv:],np.array(forces),np.array(states)


def affine_horizon(response, steps=40):
    """All input columns share the same fixed state-transition matrices."""
    A,B=response['A'],response['B'];x=np.zeros(A.shape[0]);derivative=np.zeros_like(B)
    states=[];state_inputs=[];loads=[];load_inputs=[]
    for _ in range(steps):
        loads.append(response['force']+response['Lx']@x)
        load_inputs.append(response['Lx']@derivative+response['Lu'])
        x=A@x+response['offset'];derivative=A@derivative+B
        states.append(x.copy());state_inputs.append(derivative.copy())
    return np.array(states),np.array(state_inputs),np.array(loads),np.array(load_inputs)


def piecewise_propagate(model, data, tangent, response, delta, steps=40):
    """Diagnostic: update friction branches, while geometry stays fixed.

    The same convex local force law is solved each step. New contact points
    cannot appear here. This is not adopted in the runtime controller.
    """
    m,d=model,data;n=m.nv;h=m.opt.timestep
    M,J=tangent['mass'],tangent['jacobian']
    rq=response['rigid_residual_wrt_position'];rv=response['rigid_residual_wrt_velocity']
    drive=tangent['moment'].T*(m.actuator_gainprm[:,0]*tangent['free_drives'])
    b0=d.qfrc_smooth+d.efm_c+d.efm_ca;reference=d.efc_aref.copy()
    q=response['q0'].copy();v=response['v0'].copy();dq=np.zeros(n)
    acceleration=d.qacc.copy();forces=[];states=[];iterations=[]
    cost=np.empty(1)
    for _ in range(steps):
        dv=v-response['v0']
        b=b0-rq@dq-rv@dv+drive@delta
        aref=reference-response['position_coefficient']*(J@dq)-response['velocity_coefficient']*(J@dv)
        for iteration in range(30):
            jar=J@acceleration-aref;C=contact_hessian(m,d,jar)
            residual=M@acceleration-b-J.T@d.efc_force
            if abs(residual).max(initial=0)<1e-9:break
            change=cho_solve(cho_factor(M+J.T@C@J,lower=True),-residual)
            mujoco.mj_constraintUpdate(m,d,jar,cost,0)
            value=.5*acceleration@M@acceleration-b@acceleration+cost[0]
            accepted=False
            for alpha in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
                trial=acceleration+alpha*change
                mujoco.mj_constraintUpdate(m,d,J@trial-aref,cost,0)
                score=.5*trial@M@trial-b@trial+cost[0]
                if score<=value+1e-12:
                    acceleration=trial;accepted=True;break
            if not accepted:raise ValueError('piecewise contact line search failed')
        else:raise ValueError('piecewise contact iteration budget exhausted')
        forces.append(d.efc_force.copy());iterations.append(iteration)
        v+=h*acceleration;mujoco.mj_integratePos(m,q,v,h)
        mujoco.mj_differentiatePos(m,dq,1.,response['q0'],q)
        states.append(np.r_[dq,v-response['v0']])
    return q,v,np.array(forces),np.array(states),iterations
