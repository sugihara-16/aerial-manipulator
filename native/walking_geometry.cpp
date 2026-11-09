#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <map>
using Mat=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using M3=Eigen::Matrix<double,3,3,Eigen::RowMajor>;
using V3=Eigen::Vector3d;
static M3 skew(const V3& v){M3 S;S<<0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0;return S;}
extern "C" void walking_geometry(const mjModel* m,mjData* d,int nd,const int* indices,const double* x,const double* base,
 int nf,const int* foot_bodies,const int* offsets,const double* vertices,const double* centers,const double* targets,
 const double* rotations,const int* aligned,const int* support,const double* com,int nb,const int* bodies,const double* bound_centers,const double* halves,
 double* residual,double* jacobian){
 const int nv=m->nv,rows=12*nf+nd+2+nb+1;
 Eigen::Map<Eigen::VectorXd> r(residual,rows);Eigen::Map<Mat> J(jacobian,rows,nd);r.setZero();J.setZero();
 std::copy(base,base+m->nq,d->qpos);std::vector<double> delta(nv,0.);
 for(int i=0;i<nd;++i)delta[indices[i]]=x[i];mj_integratePos(m,d->qpos,delta.data(),1.);
 mj_kinematics(m,d);mj_comPos(m,d);
 Mat T=Mat::Identity(nd,nd);V3 rv=Eigen::Map<const V3>(x+3);double angle=rv.norm();M3 K=skew(rv);
 if(angle>1e-7)T.block<3,3>(3,3)=M3::Identity()-(1-std::cos(angle))/(angle*angle)*K+(angle-std::sin(angle))/(angle*angle*angle)*K*K;
 Mat jp(3,nv),jr(3,nv),jb(3,nv);int row=0;
 auto select=[&](const Mat& A){Mat B(A.rows(),nd);for(int j=0;j<nd;++j)B.col(j)=A.col(indices[j]);B.middleCols(3,3)=(B.middleCols(3,3)*T.block<3,3>(3,3)).eval();return B;};
 for(int f=0;f<nf;++f){int b=foot_bodies[f];M3 R=Eigen::Map<const M3>(d->xmat+9*b);V3 p=Eigen::Map<const V3>(d->xpos+3*b);
  V3 center=p+R*Eigen::Map<const V3>(centers+3*f),lowest;double z=1e100;
  for(int k=offsets[f];k<offsets[f+1];++k){V3 v=p+R*Eigen::Map<const V3>(vertices+3*k);if(v[2]<z){z=v[2];lowest=v;}}
  mj_jac(m,d,jp.data(),jr.data(),center.data(),b);Mat point_jac=jp,Jr=select(jr);
  mj_jac(m,d,jp.data(),nullptr,lowest.data(),b);point_jac.row(2)=jp.row(2);Mat Jp=select(point_jac);
  V3 point=center;point[2]=z;V3 w(30.,30.,support[f]?300.:30.);
  r.segment<3>(row)=w.asDiagonal()*(point-Eigen::Map<const V3>(targets+3*f));J.block(row,0,3,nd)=w.asDiagonal()*Jp;row+=3;
  M3 Rt=Eigen::Map<const M3>(rotations+9*f);
  for(int a=0;a<3;++a){double weight=aligned[f]?(a==0?(.2+(support[f]?29.8:0.)):(support[f]?1.:.2)):.2;V3 goal=aligned[f]&&a==0?V3(0,0,-1):V3(Rt.col(a));r.segment<3>(row)=weight*(R.col(a)-goal);J.block(row,0,3,nd)=-weight*skew(R.col(a))*Jr;row+=3;}
 }
 const double regbase[]={1.,.5,.25,.12,.12,.12};
 for(int j=0;j<nd;++j){double w=j<6?regbase[j]:.008;r[row+j]=w*x[j];J(row+j,j)=w;}row+=nd;
 mj_jacSubtreeCom(m,d,jp.data(),0);r.segment<2>(row)=3*(Eigen::Map<const Eigen::Vector2d>(d->subtree_com)-Eigen::Map<const Eigen::Vector2d>(com));J.block(row,0,2,nd)=3*select(jp).topRows(2);row+=2;
 for(int i=0;i<nb;++i){int b=bodies[i];M3 R=Eigen::Map<const M3>(d->xmat+9*b);V3 h=Eigen::Map<const V3>(halves+3*i),world=Eigen::Map<const V3>(d->xpos+3*b)+R*Eigen::Map<const V3>(bound_centers+3*i);
  double depth=.005-world[2]+R.row(2).cwiseAbs().dot(h);if(depth>0){
   mj_jac(m,d,jp.data(),jr.data(),world.data(),b);Eigen::RowVectorXd dh=Eigen::RowVectorXd::Zero(nv);
   for(int a=0;a<3;++a){double sign=(R(2,a)>0)-(R(2,a)<0);dh+=sign*h[a]*(R.col(a).cross(V3(0,0,1))).transpose()*jr;}
   Mat one=dh-jp.row(2);J.row(row)=20*select(one);r[row]=20*depth;
  }++row;
 }
 mj_collision(m,d);const int floor=mj_name2id(m,mjOBJ_GEOM,"floor");
 std::map<std::pair<int,int>,const mjContact*> pairs;
 for(int i=0;i<d->ncon;++i){const auto& c=d->contact[i];if(c.geom[0]==floor||c.geom[1]==floor||c.dist>=.001)continue;
  auto key=std::make_pair(m->geom_bodyid[c.geom[0]],m->geom_bodyid[c.geom[1]]);auto found=pairs.find(key);if(found==pairs.end()||c.dist<found->second->dist)pairs[key]=&c;}
 double squared=0.;Mat gradient=Mat::Zero(1,nd);
 for(const auto& item:pairs){const auto* c=item.second;double depth=.001-c->dist;squared+=depth*depth;
  mj_jac(m,d,jp.data(),nullptr,c->pos,item.first.first);mj_jac(m,d,jb.data(),nullptr,c->pos,item.first.second);
  Mat one=Eigen::Map<const V3>(c->frame).transpose()*(jb-jp);gradient-=depth*select(one);}
 if(squared>0.){double norm=std::sqrt(squared);r[row]=200*norm;J.row(row)=200*gradient/norm;}

}

extern "C" double walking_seed(const mjModel* m,mjData* d,int nd,const int* indices,const int* qi,const double* limits,
 double* q,const double* reference,int nf,const int* foot_bodies,const int* offsets,const double* vertices,const double* centers,const double* targets,const double* rotations,const int* aligned,
 int nb,const int* bodies,const double* bound_centers,const double* halves,int iterations){
 const int nv=m->nv;const int floor=mj_name2id(m,mjOBJ_GEOM,"floor");
 Mat jp(3,nv),jr(3,nv),jb(3,nv);std::vector<double> delta(nv),dq(nv);const double weights[]={1.,.5,.25,.12,.12,.12};
 auto select=[&](const Mat& a){Mat b(a.rows(),nd);for(int j=0;j<nd;++j)b.col(j)=a.col(indices[j]);return b;};
 for(int iter=0;iter<iterations;++iter){
  std::copy(q,q+m->nq,d->qpos);mj_kinematics(m,d);mj_comPos(m,d);mj_collision(m,d);mj_differentiatePos(m,delta.data(),1.,q,reference);
  Mat A=Mat::Zero(6*nf+6+nd+nb+d->ncon,nd);Eigen::VectorXd rhs=Eigen::VectorXd::Zero(A.rows());int row=0;
  Mat ang(3*nf,nd);Eigen::VectorXd angle_error(3*nf);
  for(int f=0;f<nf;++f){int b=foot_bodies[f];M3 R=Eigen::Map<const M3>(d->xmat+9*b),Rt=Eigen::Map<const M3>(rotations+9*f);V3 p=Eigen::Map<const V3>(d->xpos+3*b),center=p+R*Eigen::Map<const V3>(centers+3*f),lowest;double z=1e100;
   for(int k=offsets[f];k<offsets[f+1];++k){V3 v=p+R*Eigen::Map<const V3>(vertices+3*k);if(v[2]<z){z=v[2];lowest=v;}}
   mj_jac(m,d,jp.data(),jr.data(),center.data(),b);Mat point=jp;ang.block(3*f,0,3,nd)=select(jr);V3 err=V3::Zero();for(int j=0;j<3;++j)err+=R.col(j).cross(Rt.col(j));M3 tilt=M3::Identity()-R.col(0)*R.col(0).transpose();double nw=targets[3*f+2]<1e-6?30.:.2;ang.block(3*f,0,3,nd)=(nw*tilt+1.*(M3::Identity()-tilt))*select(jr);angle_error.segment<3>(3*f)=nw*R.col(0).cross(V3(0,0,-1))+.5*(M3::Identity()-tilt)*err;
   if(!aligned[f]){ang.block(3*f,0,3,nd)=.2*select(jr);angle_error.segment<3>(3*f)=.1*err;}
   mj_jac(m,d,jp.data(),nullptr,lowest.data(),b);point.row(2)=jp.row(2);center[2]=z;A.block(row,0,3,nd)=10*select(point);rhs.segment<3>(row)=10*(Eigen::Map<const V3>(targets+3*f)-center);row+=3;
  }
  A.block(row,0,3*nf,nd)=ang;rhs.segment(row,3*nf)=angle_error;row+=3*nf;
  for(int i=0;i<6;++i){for(int j=0;j<nd;++j)if(indices[j]==i)A(row+i,j)=weights[i];rhs[row+i]=weights[i]*delta[i];}row+=6;
  for(int j=0;j<nd;++j){A(row+j,j)=.008;rhs[row+j]=.008*delta[indices[j]];}row+=nd;
  for(int i=0;i<nb;++i){int b=bodies[i];M3 R=Eigen::Map<const M3>(d->xmat+9*b);V3 h=Eigen::Map<const V3>(halves+3*i),world=Eigen::Map<const V3>(d->xpos+3*b)+R*Eigen::Map<const V3>(bound_centers+3*i);
   double depth=.005-world[2]+R.row(2).cwiseAbs().dot(h);if(depth<=0)continue;
   mj_jac(m,d,jp.data(),jr.data(),world.data(),b);Eigen::RowVectorXd dh=Eigen::RowVectorXd::Zero(nv);
   for(int j=0;j<3;++j){double sign=(R(2,j)>0)-(R(2,j)<0);dh+=sign*h[j]*R.col(j).cross(V3(0,0,1)).transpose()*jr;}
   Mat one=jp.row(2)-dh;A.row(row)=20*select(one);rhs[row]=20*depth;++row;
  }
  std::map<std::pair<int,int>,const mjContact*> pairs;
  for(int i=0;i<d->ncon;++i){const auto& c=d->contact[i];if(c.geom[0]==floor||c.geom[1]==floor||c.dist>=.002)continue;auto key=std::make_pair(m->geom_bodyid[c.geom[0]],m->geom_bodyid[c.geom[1]]);auto found=pairs.find(key);if(found==pairs.end()||c.dist<found->second->dist)pairs[key]=&c;}
  for(const auto& item:pairs){auto c=item.second;mj_jac(m,d,jp.data(),nullptr,c->pos,item.first.first);mj_jac(m,d,jb.data(),nullptr,c->pos,item.first.second);Mat one=Eigen::Map<const V3>(c->frame).transpose()*(jb-jp);A.row(row)=20*select(one);rhs[row]=20*(.002-c->dist);++row;}
  Eigen::ColPivHouseholderQR<Mat> qr(A.topRows(row));qr.setThreshold(1e-8);Eigen::VectorXd step=qr.solve(rhs.head(row));step*=std::min(1.,.15/std::max(step.cwiseAbs().maxCoeff(),1e-9));std::fill(dq.begin(),dq.end(),0.);for(int j=0;j<nd;++j)dq[indices[j]]=step[j];mj_integratePos(m,q,dq.data(),1.);
  for(int j=0;j<nd-6;++j)q[qi[j]]=std::clamp(q[qi[j]],limits[2*j],limits[2*j+1]);if(step.cwiseAbs().maxCoeff()<1e-6)break;
 }
 std::copy(q,q+m->nq,d->qpos);mj_kinematics(m,d);double error=0.;
 for(int f=0;f<nf;++f){int b=foot_bodies[f];M3 R=Eigen::Map<const M3>(d->xmat+9*b);V3 p=Eigen::Map<const V3>(d->xpos+3*b),center=p+R*Eigen::Map<const V3>(centers+3*f);double z=1e100;for(int k=offsets[f];k<offsets[f+1];++k)z=std::min(z,(p+R*Eigen::Map<const V3>(vertices+3*k))[2]);center[2]=z;error=std::max(error,(center-Eigen::Map<const V3>(targets+3*f)).norm());}
 return error;
}

// One new branch's finite hinge motion, before a complete tree exists.
extern "C" void walking_leaf_motion(const double* seams,const double* pads,const double* axes,const double* origins,
 const double* parent,int outgoing,int incoming,int pad,const double* initial,const double* velocity,double period,double duty,
 double height,int samples,double* report,double* configurations){
 using M4=Eigen::Matrix<double,4,4,Eigen::RowMajor>;M4 P=Eigen::Map<const M4>(parent),mate=M4::Identity();mate(0,0)=mate(1,1)=-1.;
 V3 x=Eigen::Map<const V3>(initial),x0=x,vel=Eigen::Map<const V3>(velocity);Mat J(3,3);V3 point;
 auto evaluate=[&](){M4 frames[3];const int ports[]={outgoing,incoming,pad};
  for(int k=0;k<3;++k){int p=ports[k];V3 axis=Eigen::Map<const V3>(axes+3*p),origin=Eigen::Map<const V3>(origins+3*p);M3 S=skew(axis),R=M3::Identity()+std::sin(x[k])*S+(1-std::cos(x[k]))*S*S;
   M4 F=M4::Identity();F.block<3,3>(0,0)=R;F.block<3,1>(0,3)=origin-R*origin;frames[k]=F*Eigen::Map<const M4>((k==2?pads:seams)+16*p);}
  M4 inverse=M4::Identity();inverse.block<3,3>(0,0)=frames[1].block<3,3>(0,0).transpose();inverse.block<3,1>(0,3)=-inverse.block<3,3>(0,0)*frames[1].block<3,1>(0,3);
  M4 child=P*frames[0]*mate*inverse,foot=child*frames[2];point=foot.block<3,1>(0,3);
  for(int k=0;k<3;++k){const M4& owner=k==0?P:child;int p=ports[k];V3 axis=owner.block<3,3>(0,0)*Eigen::Map<const V3>(axes+3*p),origin=owner.block<3,1>(0,3)+owner.block<3,3>(0,0)*Eigen::Map<const V3>(origins+3*p);if(k==1)axis=-axis;J.col(k)=axis.cross(point-origin);}
 };
 evaluate();V3 p0=point,first,previous;double peak=0.,errors=0.,change=0.,speed=0.;
 for(int k=0;k<=samples;++k){double phase=std::fmod(.5*duty+double(k)/samples,1.),s=std::clamp((phase-duty)/(1-duty),0.,1.),blend=s*s*s*(10-15*s+6*s*s);V3 target=p0+vel*period*(.5*duty-phase+blend);target[2]+=64*height*std::pow(s,3)*std::pow(1-s,3);
  for(int j=0;j<12;++j){evaluate();V3 delta=(J.transpose()*J+1e-5*M3::Identity()).ldlt().solve(-J.transpose()*(point-target)-1e-5*(x-x0));x+=delta*std::min(1.,.12/std::max(delta.cwiseAbs().maxCoeff(),1e-9));x=x.cwiseMax(-M_PI/2+.025).cwiseMin(M_PI/2-.025);if(delta.cwiseAbs().maxCoeff()<1e-5)break;}
  evaluate();double error=(point-target).norm();peak=std::max(peak,error);errors+=error*error;if(k==0)first=x;else{change+=(x-previous).squaredNorm();speed=std::max(speed,(x-previous).cwiseAbs().maxCoeff()*samples/period);}previous=x;Eigen::Map<V3>(configurations+3*k)=x;
 }
 double closure=(x-first).norm();report[0]=4*peak*peak+errors/(samples+1)+.03*.03*closure*closure+.002*.002*change/(3*samples);report[1]=peak;report[2]=closure;report[3]=speed;
}

// Same part-clearance and deepest non-floor contact rows as support posture.
extern "C" void walking_posture_clearance(const mjModel* m,mjData* d,int nb,const int* bodies,
 const double* centers,const double* halves,double* residual,double* jacobian){
 const int nv=m->nv;Eigen::Map<Eigen::VectorXd> r(residual,nb+1);Eigen::Map<Mat> J(jacobian,nb+1,nv);r.setZero();J.setZero();
 mj_kinematics(m,d);mj_comPos(m,d);Mat jp(3,nv),jr(3,nv),jb(3,nv);
 for(int i=0;i<nb;++i){int b=bodies[i];M3 R=Eigen::Map<const M3>(d->xmat+9*b);V3 h=Eigen::Map<const V3>(halves+3*i),world=Eigen::Map<const V3>(d->xpos+3*b)+R*Eigen::Map<const V3>(centers+3*i);
  double depth=.03-world[2]+R.row(2).cwiseAbs().dot(h);if(depth<=0)continue;
  mj_jac(m,d,jp.data(),jr.data(),world.data(),b);Eigen::RowVectorXd dh=Eigen::RowVectorXd::Zero(nv);
  for(int a=0;a<3;++a){double sign=(R(2,a)>0)-(R(2,a)<0);dh+=sign*h[a]*(R.col(a).cross(V3(0,0,1))).transpose()*jr;}
  J.row(i)=200*(dh-jp.row(2));r[i]=200*depth;
 }
 mj_collision(m,d);const int floor=mj_name2id(m,mjOBJ_GEOM,"floor");double deepest=0.;const mjContact* chosen=nullptr;
 for(int i=0;i<d->ncon;++i){const auto& c=d->contact[i];if(c.geom[0]==floor||c.geom[1]==floor||c.dist>=.002)continue;if(.002-c.dist>deepest){deepest=.002-c.dist;chosen=&c;}}
 if(chosen){int a=m->geom_bodyid[chosen->geom[0]],b=m->geom_bodyid[chosen->geom[1]];mj_jac(m,d,jp.data(),nullptr,chosen->pos,a);mj_jac(m,d,jb.data(),nullptr,chosen->pos,b);J.row(nb)=-200*Eigen::Map<const V3>(chosen->frame).transpose()*(jb-jp);r[nb]=200*deepest;}
}
