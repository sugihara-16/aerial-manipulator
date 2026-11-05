"""Periodic one-axis direction and support-force sequential convex proposal.

Only gimbal angles change. Periodic cubic knots share angle/rate constraints;
full articulated inertia and its derivatives couple these directions to
support allocations. Linearization residuals are penalized diagnostics, not
physical inputs. A separate exact force/geometry check is required afterward.
"""
import time
from types import SimpleNamespace
import numpy as np, scipy.sparse as sp, pinocchio as pin, mujoco, clarabel
from scipy.interpolate import CubicSpline
from scipy.linalg import null_space,qr
from amsrr_det.native_qp import solve_qp
from amsrr_det.walking import WalkSpec,foot_link
from amsrr_det.walking_cycle import CycleReference
from amsrr_det.walking_control import GroundControl
from amsrr_det.walking_load import motion_load,ground_contacts,support_quadrature,support_switch_times
from amsrr_det.equilibrium import point_jacobian,contact_basis
from amsrr_det.walking_gait import validate_cycle



def spline_envelope(curve, columns, left, right):
    """Exact component extrema of a piecewise polynomial of degree <= 3."""
    knots=curve.x;ids=np.flatnonzero((knots[:-1]<=right)&(knots[1:]>=left))
    lo=np.maximum(left,knots[ids])-knots[ids]
    hi=np.minimum(right,knots[ids+1])-knots[ids]
    coefficients=curve.c[:,ids][:,:,columns]
    def evaluate(t):
        out=np.zeros_like(t)
        for coefficient in coefficients:out=out*t+coefficient
        return out
    lower=np.broadcast_to(lo[:,None],coefficients.shape[1:])
    upper=np.broadcast_to(hi[:,None],coefficients.shape[1:])
    values=[evaluate(lower),evaluate(upper)]
    derivative=np.arange(len(coefficients)-1,0,-1)[:,None,None]*coefficients[:-1]
    roots=[]
    if len(derivative)==3:
        a,b,c=derivative;disc=b*b-4*a*c
        q=-.5*(b+np.copysign(np.sqrt(np.maximum(disc,0.)),b))
        valid=(disc>=0)&(a!=0)
        roots.extend([np.divide(q,a,out=np.full_like(a,np.nan),where=valid),
                      np.divide(c,q,out=np.full_like(a,np.nan),where=valid&(q!=0))])
        roots.append(np.divide(-c,b,out=np.full_like(a,np.nan),where=(a==0)&(b!=0)))
        roots.append(np.where(valid&(q==0),0.,np.nan))
    elif len(derivative)==2:
        a,b=derivative
        roots.append(np.divide(-b,a,out=np.full_like(a,np.nan),where=a!=0))
    for root in roots:
        within=(root>=lower)&(root<=upper)
        value=evaluate(np.where(within,root,lower))
        values.append(value)
    stack=np.array(values)
    return stack.min(axis=(0,1)),stack.max(axis=(0,1))


def optimize_direction(plant,record,spec,*,iterations=6,budget=5.,inspect=None):
    a=SimpleNamespace(knots=8,iterations=iterations,budget=budget,rate_fraction=.9)
    start=time.perf_counter()
    p=record;pl=plant;m=pl.mj;model=pl.pin;meta=pl.meta;data=model.createData();g={k:np.array(v)if isinstance(v,list)else v for k,v in p['gait'].items()};ref=CycleReference(pl,spec,g);links=[foot_link(*v)for v in p['surfaces']];c=GroundControl(pl,ref.pose(0),links,foot_points=g['foot_points']);dr=c.drive;T=ref.period
    nr=len(meta['rotors']);nj=dr.nj;nv=model.nv;gids=np.array([r['gimbal']for r in meta['rotors']]);mq=dr.qi[gids];mv=dr.vi[gids];pq=np.array([meta['actuators'][i]['q']for i in gids]);pv=np.array([meta['actuators'][i]['v']for i in gids]);av=np.array([v['v']for v in meta['actuators']]);limits=np.array([v['effort']for v in meta['actuators']]);maximum=np.array([r['maximum']for r in meta['rotors']]);W=meta['mass']*9.80665;H=W**1.5/np.sqrt(nr);npar=a.knots*nr
    spline=CubicSpline(np.linspace(0,T,a.knots+1),np.vstack([np.eye(a.knots),np.eye(a.knots)[0]]),bc_type='periodic')
    act=np.zeros((nv,nj));act[av,np.arange(nj)]=limits
    scale=np.full(nv,1000./W);scale[3:6]/=.5;scale[av]=1000./limits
    samples=[]
    quadrature=list(zip(*support_quadrature(T,len(links),g['duty'],g['phases'],order=3)))+[(t,0.)for t in support_switch_times(T,len(links),g['duty'],g['phases'])]
    from amsrr_det.equilibrium import static_balance
    check_times=np.linspace(0,T,97)[:-1];peaks=[]
    for t in check_times:
     q,v,acc=ref.rates(t);qp,load=motion_load(pl,dr,q,v,acc);cts=ground_contacts(c,ref.feet(t)[3]);r=static_balance(model,meta,qp,cts,generalized_load=load,fast_lp=True)
     peaks.append(r.get('minimum_peak_effort_ratio',float('inf')))
    critical=[i for i,v in enumerate(peaks)if v>.88 and v>=peaks[(i-1)%96] and v>=peaks[(i+1)%96]]
    quadrature += [(float(check_times[i]),0.)for i in sorted(critical,key=lambda i:-peaks[i])[:8]]
    def make_sample(t,w):
     q,v,acc=ref.rates(t);qp,load=motion_load(pl,dr,q,v,acc);_,vp=pl.observation(dr.d);_,_,pvi,mvi,bases=pl.coordinate_maps;pa=np.empty(nv);pa[pvi]=acc[mvi]
     for pos,vel,source in bases:
      R=pin.Quaternion(qp[pos+3:pos+7]).matrix();pa[vel:vel+3]=R.T@acc[source:source+3]-np.cross(vp[vel+3:vel+6],vp[vel:vel+3])
     contacts=ground_contacts(c,ref.feet(t)[3]);contacts=list({(ct['frame'],tuple(ct['point'])):ct for ct in contacts}.values());C=np.column_stack([point_jacobian(model,data,qp,meta['frames'][ct['frame']],ct['point']).T@contact_basis(ct)for ct in contacts]);phi=[np.kron(spline(t,k)[None],np.eye(nr))for k in range(3)];return dict(time=t,weight=w,q=qp.copy(),v=vp.copy(),a=pa,C=C,contacts=contacts,phi=phi)
    samples=[make_sample(t,w)for t,w in sorted(quadrature)]
    nt=len(samples);widths=[nj+nr+z['C'].shape[1]+nv for z in samples];offset=np.r_[npar,npar+np.cumsum(widths)];total=int(offset[-1]);X=np.zeros(npar);forces=[]
    for row in samples:
     nc=row['C'].shape[1];f=np.r_[np.zeros(nj),np.full(nr,.4/nr),np.tile([.6/(nc/3),0.,0.],nc//3)];forces.append(f)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.direct_solve_method='qdldl';settings.max_iter=100;settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-8

    def state(row,x):
     q=row['q'].copy();v=row['v'].copy();acc=row['a'].copy();q[pq]+=row['phi'][0]@x;v[pv]+=row['phi'][1]@x;acc[pv]+=row['phi'][2]@x;return q,v,acc

    def evaluate(row,x,f,derivative=False):
     q,v,acc=state(row,x);pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data);B=np.column_stack([pin.getFrameJacobian(model,data,r['frame'],pin.LOCAL_WORLD_ALIGNED)[:3].T@data.oMf[r['frame']].rotation[:,2]for r in meta['rotors']]);F=np.c_[act,B*W,row['C']*W];load=pin.rnea(model,data,q,v,acc).copy();res=scale*(load-F@f);power=sum(np.maximum(f[nj:nj+nr],0.)**1.5)*np.sqrt(nr);value=.5*res@res+power+.0005*(f[:nj]@f[:nj])
     if not derivative:return value,res,power
     if derivative=='force':return value,res,power,None,scale[:,None]*F,scale*load
     external=[pin.Force.Zero()for _ in range(model.njoints)]
     for rotor,thrust in zip(meta['rotors'],W*f[nj:nj+nr]):
      frame=model.frames[rotor['frame']];external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,thrust]),np.zeros(3)))
     dq,dv,da=pin.computeRNEADerivatives(model,data,q,v,acc,external);dq=dq.copy();dv=dv.copy();da=np.triu(da)+np.triu(da,1).T
     G=dq[:,pv]@row['phi'][0]+dv[:,pv]@row['phi'][1]+da[:,pv]@row['phi'][2]
     return value,res,power,scale[:,None]*G,scale[:,None]*F,scale*load


    def restore(x,rows=None):
     fs=[]
     for row in samples if rows is None else rows:
      nf=nj+nr+row['C'].shape[1];n=nf+nv+nr;sl=nf;en=sl+nv
      _,_,_,_,F,y=evaluate(row,x,np.zeros(nf),'force')
      if 'force_cache' not in row:
       # Directions follow one-axis trigonometric motion. A second generic
       # angle supplies the structural nonzeros hidden at the initial angle.
       _,_,_,_,Fother,_=evaluate(row,x+np.full(npar,.371),np.zeros(nf),'force')
       mask=(abs(F)>1e-12)|(abs(Fother)>1e-12);fi,fj=np.nonzero(mask)
       Aforce=sp.coo_matrix((np.ones(len(fi)),(fi,fj)),shape=(nv,n))
       mats=[Aforce+sp.coo_matrix((np.ones(nv),(np.arange(nv),sl+np.arange(nv))),shape=(nv,n))];bs=[y];cs=[clarabel.ZeroConeT(nv)]
       I=sp.eye(n,format='csc')[:nj+nr];mats.extend([I,-I]);bs.extend([np.r_[np.full(nj,.88),maximum/W],np.r_[np.full(nj,.88),np.zeros(nr)]]);cs.extend([clarabel.NonnegativeConeT(nj+nr)]*2)
       for j in range(row['C'].shape[1]//3):
        k=nj+nr+3*j;facets=np.array([[-.8,1.,1.],[-.8,1.,-1.],[-.8,-1.,1.],[-.8,-1.,-1.]]);mats.append(sp.coo_matrix((facets.ravel(),(np.repeat(np.arange(4),3),np.tile(np.arange(k,k+3),4))),shape=(4,n)));bs.append(np.zeros(4));cs.append(clarabel.NonnegativeConeT(4))
       for j in range(nr):
        mats.append(sp.coo_matrix(([-1.,-1.],([0,2],[en+j,nj+j])),shape=(3,n)));bs.append(np.array([0.,1.,0.]));cs.append(clarabel.PowerConeT(2/3))
       A=sp.vstack(mats,format='csc');b=np.concatenate(bs);columns=np.repeat(np.arange(n),np.diff(A.indptr));slots=np.flatnonzero((A.indices<nv)&(columns<nf));ri=A.indices[slots];ci=columns[slots];A.data[slots]=F[ri,ci]
       diag=np.r_[np.full(nj,.001),np.zeros(nf-nj),np.ones(nv),np.zeros(nr)];obj=np.r_[np.zeros(en),np.full(nr,np.sqrt(nr))]
       solver=clarabel.DefaultSolver(sp.diags(diag,format='csc'),obj,A,b,cs,settings);row['force_cache']=(solver,A,b,slots,ri,ci,mask)
      else:
       solver,A,b,slots,ri,ci,mask=row['force_cache']
       if np.max(abs(F[~mask]),initial=0.)>1e-9:raise RuntimeError('force balance sparsity changed')
       A.data[slots]=F[ri,ci];b[:nv]=y;solver.update(A=A,b=b)
      sol=solver.solve()
      if str(sol.status)not in ['Solved','AlmostSolved']:return None
      f=np.array(sol.x[:nf]);row['_restored']=(x.copy(),y-F@f);fs.append(f)
     return fs

    def cost(x,fs):
     reports=[]
     for z,f in zip(samples,fs):
      cached=z.get('_restored')
      if cached is not None and np.array_equal(cached[0],x):
       res=cached[1];power=sum(np.maximum(f[nj:nj+nr],0.)**1.5)*np.sqrt(nr)
       reports.append((.5*res@res+power+.0005*(f[:nj]@f[:nj]),res,power))
      else:reports.append(evaluate(z,x,f))
     return sum(max(z['weight'],.001)*r[0]for z,r in zip(samples,reports)),sum(z['weight']*r[2]for z,r in zip(samples,reports)),max(np.linalg.norm(r[1])/1000 for r in reports)
    # Dense linear angle/rate checks constrain the actual periodic interpolant.
    angle_rows=[];angle_limits=[]
    lo=np.array([meta['actuators'][i]['lower']+.01 for i in gids]);hi=np.array([meta['actuators'][i]['upper']-.01 for i in gids]);speed=a.rate_fraction*np.array([meta['actuators'][i]['speed'] for i in gids])
    def envelope(t0,t1,derivative):
     return spline_envelope(ref.spline.derivative(derivative),mq,t0,t1)
    for left,right in zip(np.linspace(0,T,a.knots+1)[:-1],np.linspace(0,T,a.knots+1)[1:]):
     h=right-left;p0=spline(left);p1=spline(right);v0=spline(left,1);v1=spline(right,1)
     coefficients=np.array([p0,p0+h*v0/3,p1-h*v1/3,p1]);derivatives=3*np.diff(coefficients,axis=0)/h
     low,high=envelope(left,right,0);vlow,vhigh=envelope(left,right,1)
     for coefficient in coefficients:
      P=np.kron(coefficient[None],np.eye(nr));angle_rows.extend([P,-P]);angle_limits.extend([hi-high,low-lo])
     for coefficient in derivatives:
      V=np.kron(coefficient[None],np.eye(nr));angle_rows.extend([V,-V]);angle_limits.extend([speed-vhigh,speed+vlow])
    Aangle=sp.csc_matrix(np.vstack(angle_rows));bangle=np.concatenate(angle_limits);history=[];forces=restore(X)
    if forces is None:raise ValueError('direction_force_initialization')
    previous,previous_power,previous_error=cost(X,forces)
    def proposed_gait():
     # Both periodic cubics have knots on this common grid. Keep the original
     # body/Dock knots exactly when the existing grid already contains it.
     count=int(np.lcm(len(g['configurations'])-1,a.knots))+1
     qvalues=np.array(g['configurations'],copy=True) if count==len(g['configurations']) else np.array([ref.pose(t)for t in np.linspace(0,T,count)])
     for t,q in zip(np.linspace(0,T,count),qvalues):q[mq]+=np.kron(spline(t)[None],np.eye(nr))@X
     return dict(g,foot_links=links,configurations=qvalues)
    for iteration in range(a.iterations):
     remaining=a.budget-(time.perf_counter()-start)
     if remaining<=.1:break
     Hessian=np.eye(npar)*.001;gradient=np.zeros(npar)
     for row,f in zip(samples,forces):
      _,res,_,G,F,y=evaluate(row,X,f,True);nf=len(f);active=[]
      for j in range(nj+nr):
       lower=-.88 if j<nj else 0.;upper=.88 if j<nj else maximum[j-nj]/W
       if min(f[j]-lower,upper-f[j])<1e-6:active.append(np.eye(nf)[j])
      for j in range(row['C'].shape[1]//3):
       off=nj+nr+3*j;facets=np.array([[-.8,1.,1.],[-.8,1.,-1.],[-.8,-1.,1.],[-.8,-1.,-1.]])
       for facet in facets:
        if facet@f[off:off+3]>-1e-6:
         axis=np.zeros(nf);axis[off:off+3]=facet;active.append(axis)
      N=null_space(np.array(active)) if active else np.eye(nf)
      curvature=np.r_[np.full(nj,.001),.75*np.sqrt(nr)/np.sqrt(np.maximum(f[nj:nj+nr],1e-8)),np.full(nf-nj-nr,1e-9)]
      D=np.vstack([G,np.zeros((nf,npar))])
      if N.shape[1]:
       B=np.vstack([-F,np.diag(np.sqrt(curvature))])@N
       Q,R,piv=qr(B,mode='economic',pivoting=True);rank=np.sum(abs(np.diag(R))>1e-8);Q=Q[:,:rank]
       D-=Q@(Q.T@D)
      weight=max(row['weight'],.001)
      Hessian+=weight*(D.T@D);gradient+=weight*(G.T@res)
     constraint=np.vstack([np.eye(npar),-np.eye(npar),-Aangle.toarray()])
     bound=np.r_[np.full(2*npar,-.3),Aangle@X-bangle]
     try:dx=solve_qp((Hessian+Hessian.T)*.5,-gradient,constraint.T,bound)[0]
     except ValueError as error:history.append(dict(iteration=iteration,status=str(error)));break
     accepted=False
     value,power,error=previous,previous_power,previous_error
     for step in [1.,.5,.25]:
      trial=X+step*dx;ff=restore(trial)
      if ff is None:continue
      value,power,error=cost(trial,ff)
      # A lower energy score must not buy a worse force imbalance. Once
      # restored, keep restoration within numerical tolerance; dense physical
      # acceptance still runs independently after these proposal updates.
      if value<previous-1e-8 and error<=max(1e-4,.95*previous_error):
       X=trial;forces=ff;previous=value;previous_power=power;previous_error=error;accepted=True;break
     history.append(dict(iteration=iteration,accepted=accepted,step=step,cost=value,power=power,force_error=error,seconds=time.perf_counter()-start))
     if iteration in [1,3]:
      current_peaks=[]
      for t in check_times:
       q,v,acc=ref.rates(t);q[mq]+=np.kron(spline(t)[None],np.eye(nr))@X;v[mv]+=np.kron(spline(t,1)[None],np.eye(nr))@X;acc[mv]+=np.kron(spline(t,2)[None],np.eye(nr))@X
       qp,load=motion_load(pl,dr,q,v,acc);cts=ground_contacts(c,ref.feet(t)[3]);r=static_balance(model,meta,qp,cts,generalized_load=load,fast_lp=True)
       current_peaks.append(r.get('minimum_peak_effort_ratio',float('inf')))
      critical=[i for i,value in enumerate(current_peaks)if value>.86 and value>=current_peaks[(i-1)%96]and value>=current_peaks[(i+1)%96]]
      added=[make_sample(float(check_times[i]),0.)for i in sorted(critical,key=lambda i:-current_peaks[i])[:8] if min(abs(z['time']-check_times[i])for z in samples)>1e-8]
      if added:
       restored=restore(X,added)
       if restored is not None:
        samples.extend(added);forces.extend(restored);previous,previous_power,previous_error=cost(X,forces);accepted=True
     if inspect is not None and (iteration%2==1 or not accepted or iteration==a.iterations-1):
      inspect(proposed_gait(),iteration+1)
     if not accepted:break
    return proposed_gait(),dict(seconds=time.perf_counter()-start,variables=total,angle_variables=npar,samples=nt,cost=cost(X,forces),history=history,method='condensed_force_tangent_periodic_direction')
