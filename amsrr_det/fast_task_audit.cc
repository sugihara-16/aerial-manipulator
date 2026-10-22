// Read-only reduction of the same per-step evaluation quantities as TaskAudit.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <vector>
extern "C" int audit_version(){return mjVERSION_HEADER;}
extern "C" int audit_step(const mjModel* m,const mjData* d,int carried,int grasp,
 int object,int pads,const int* bodies,const int* group,const int* support,
 int na,const int* ai,const int* qi,const int* vi,const double* limits,const double* effort,
 int shape,const double* size,int nv,const double* vertices,double top,
 double* stats,double* saturation,double* gaps,double* anchors,double* normal,int* unexpected){
 const int ob=m->geom_bodyid[object];const double* pos=d->xpos+3*ob;const double* R=d->xmat+9*ob;
 double radius=0.;
 if(shape==0)for(int j=0;j<3;++j)radius+=std::abs(R[6+j])*size[j]/2.;
 else if(shape==1){for(int j=0;j<3;++j)radius+=std::pow(R[6+j]*size[j]/2.,2);radius=std::sqrt(radius);}
 else if(shape==2)radius=size[0]/2.*std::hypot(R[6],R[7])+size[2]/2.*std::abs(R[8]);
 else {radius=-1e100;for(int k=0;k<nv;++k)radius=std::max(radius,-mju_dot3(R+6,vertices+3*k));}
 const double clearance=pos[2]-radius-top;
 stats[6]=std::max(stats[6],clearance);if(carried)stats[7]=std::min(stats[7],clearance);
 for(int j=0;j<na;++j){
  const int a=ai[j];double raw=m->actuator_gainprm[a*mjNGAIN]*d->ctrl[a]+m->actuator_biasprm[a*mjNBIAS+1]*d->actuator_length[a]+m->actuator_biasprm[a*mjNBIAS+2]*d->actuator_velocity[a];
  stats[3]=std::max(stats[3],std::abs(d->qvel[vi[j]]));stats[4]=std::max(stats[4],std::abs(d->actuator_force[a])/effort[j]);
  stats[5]=std::max({stats[5],limits[2*j]-d->qpos[qi[j]],d->qpos[qi[j]]-limits[2*j+1]});
  stats[8]=std::max(stats[8],std::abs(raw)/effort[j]);saturation[j]+=(std::abs(raw)>effort[j])*m->opt.timestep;
  stats[9]=std::max(stats[9],std::abs(d->ctrl[a]-d->actuator_length[a]));
 }
 // Normal force, world force/moment, and weighted object-local pressure center.
 std::vector<double> sums(10*pads,0.);int count=0;
 for(int k=0;k<d->ncon;++k){
  const auto& c=d->contact[k];int a=c.geom[0],b=c.geom[1];bool first=a==object,second=b==object;int other=first?b:a,g=group[other];
  stats[2]=std::max(stats[2],-c.dist);
  if(c.dist < -1e-5 && ((!first&&!second)||((first||second)&&(g<0||!grasp)&&!support[other]))){unexpected[2*count]=a;unexpected[2*count+1]=b;++count;}
  if((!first&&!second)||g<0||c.efc_address<0)continue;
  if(c.dim!=3)return -1;
  const double* local=d->efc_force+c.efc_address;double force[3],arm[3],moment[3],relative[3],center[3];
  mju_mulMatTVec(force,c.frame,local,3,3);mju_scl3(force,force,first?1.:-1.);
  mju_sub3(arm,c.pos,d->xpos+3*bodies[g]);mju_cross(moment,arm,force);
  mju_sub3(relative,c.pos,pos);mju_mulMatTVec(center,R,relative,3,3);
  double w=std::max(0.,local[0]);double* s=sums.data()+10*g;s[0]+=w;
  for(int j=0;j<3;++j){s[1+j]+=force[j];s[4+j]+=moment[j];s[7+j]+=w*center[j];}
 }
 for(int k=0;k<pads;++k){
  const double* s=sums.data()+10*k;normal[k]=s[0];stats[0]=std::max(stats[0],mju_norm3(s+1));stats[1]=std::max(stats[1],mju_norm3(s+4));
  if(s[0]>.05)stats[13]=d->time;
  if(!carried)continue;
  gaps[k]=s[0]>.5?0.:gaps[k]+m->opt.timestep;stats[12]=std::max(stats[12],gaps[k]);
  if(s[0]<=.5)continue;
  double center[3];mju_scl3(center,s+7,1./s[0]);double* anchor=anchors+10*k;
  const double* p=d->xpos+3*bodies[k];const double* T=d->xmat+9*bodies[k];
  if(!anchor[0]){
   anchor[0]=1.;std::copy_n(center,3,anchor+1);std::copy_n(center,3,anchor+7);
   double world[3],relative[3];mju_mulMatVec(world,R,center,3,3);mju_addTo3(world,pos);mju_sub3(relative,world,p);mju_mulMatTVec(anchor+4,T,relative,3,3);
  }
  double delta[3];mju_sub3(delta,center,anchor+1);stats[10]=std::max(stats[10],mju_norm3(delta));
  double world[3],relative[3],current[3];mju_mulMatVec(world,T,anchor+4,3,3);mju_addTo3(world,p);mju_sub3(relative,world,pos);mju_mulMatTVec(current,R,relative,3,3);
  mju_sub3(delta,current,anchor+7);stats[11]=std::max(stats[11],mju_norm3(delta));
 }
 return count;
}

extern "C" void motion_maximum(int states,int nq,int rows,int hinges,const int* qi,
 const double* levers,const int* roots,const double* reaches,const double* q,const double* anchor,double* output){
 std::vector<double> delta(hinges);
 for(int i=0;i<states;++i){const double* state=q+i*nq;output[i]=0.;
  for(int j=0;j<hinges;++j)delta[j]=std::abs(state[qi[j]]-anchor[qi[j]]);
  // Only one or two free roots occur; cache their distance per state.
  std::vector<double> translation(nq,-1.),rotation(nq);
  for(int r=0;r<rows;++r){double movement=0.;for(int j=0;j<hinges;++j)movement+=levers[r*hinges+j]*delta[j];
   int k=roots[r];if(k>=0){
    if(translation[k]<0.){double t=0.,minus=0.,plus=0.;
     for(int j=0;j<3;++j)t+=std::pow(state[k+j]-anchor[k+j],2);
     for(int j=3;j<7;++j){minus+=std::pow(state[k+j]-anchor[k+j],2);plus+=std::pow(state[k+j]+anchor[k+j],2);}
     translation[k]=std::sqrt(t);rotation[k]=std::min(2.,2.*std::sqrt(std::min(minus,plus)));
    }
    movement+=translation[k]+rotation[k]*reaches[r];
   }
   output[i]=std::max(output[i],movement);
  }
 }
}
