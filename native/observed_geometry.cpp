// Exact MuJoCo surface queries and Jacobians, batched across Python boundary.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
extern "C" int observed_geometry_version(){return mjVERSION_HEADER;}
// Same world-frame PD terms as FastHold, without per-module Python arrays.
extern "C" void hold_acceleration(int count,const int* bq,const int* bv,
    const double* q0,const double* v0,const double* q,const double* v,
    const double* delta,double* out){
  std::fill(out,out+6,0.);
  for(int i=0;i<count;++i){int p=bq[i],j=bv[i];double R[9],R0[9],e[3],w[3],w0[3];
    mju_quat2Mat(R,q+p+3);mju_quat2Mat(R0,q0+p+3);
    mju_mulMatVec3(e,R0,delta+j+3);mju_mulMatVec3(w,R,v+j+3);mju_mulMatVec3(w0,R0,v0+j+3);
    for(int k=0;k<3;++k){out[k]-=4.*(delta[j+k]+v[j+k]-v0[j+k]);out[k+3]-=9.*e[k]+6.*(w[k]-w0[k]);}
  }
  for(int k=0;k<6;++k)out[k]/=count;
}
extern "C" void contact_mechanics(const mjModel* m,mjData* d,int nj,const int* vi,
    int nr,const int* sites,double* mass,double* action){
  if(m->ntendon||m->nflex){mj_fwdPosition(m,d);mj_fwdVelocity(m,d);}
  else {
    mj_kinematics(m,d);mj_comPos(m,d);mj_crb(m,d);mj_collision(m,d);
    mj_comVel(m,d);mj_rne(m,d,0,d->qfrc_bias);
  }
  mj_fullM(m,d,mass);
  int nu=nj+nr;std::fill(action,action+m->nv*nu,0.);
  for(int j=0;j<nj;++j)action[vi[j]*nu+j]=1.;
  std::vector<double> jac(3*m->nv);
  for(int r=0;r<nr;++r){int s=sites[r];mj_jacSite(m,d,jac.data(),nullptr,s);
    const double* R=d->site_xmat+9*s;
    for(int j=0;j<m->nv;++j)action[j*nu+nj+r]=R[2]*jac[j]+R[5]*jac[m->nv+j]+R[8]*jac[2*m->nv+j];
  }
}
extern "C" int pad_gap_rows(const mjModel* m,mjData* d,int obj,int np,const int* starts,
                            const int* geoms,double* gaps,double* rows){
  std::vector<int> membership(m->ngeom,-1),active(np,-1);
  for(int p=0;p<np;++p)for(int j=starts[p];j<starts[p+1];++j)membership[geoms[j]]=p;
  for(int i=0;i<d->ncon;++i){const mjContact& c=d->contact[i];
    int other=c.geom[0]==obj?c.geom[1]:(c.geom[1]==obj?c.geom[0]:-1);
    int p=other<0?-1:membership[other];
    if(p>=0&&(active[p]<0||c.dist<d->contact[active[p]].dist))active[p]=i;
  }
  std::vector<double> ja(3*m->nv),jb(3*m->nv);
  for(int p=0;p<np;++p){
    int a=-1,b=obj;double pa[3],pb[3],normal[3],gap=std::numeric_limits<double>::infinity();
    if(active[p]>=0){const mjContact& c=d->contact[active[p]];a=c.geom[0];b=c.geom[1];gap=c.dist;
      std::copy(c.pos,c.pos+3,pa);std::copy(c.pos,c.pos+3,pb);std::copy(c.frame,c.frame+3,normal);
    }else{
      double segment[6],nearest[6]={};
      for(int j=starts[p];j<starts[p+1];++j){int g=geoms[j];
        double distance=mj_geomDistance(m,d,g,obj,gap>0?std::min(1.,gap):1.,segment);
        if(distance<gap){gap=distance;a=g;std::copy(segment,segment+6,nearest);}
      }
      if(gap>=1.){
        gap=std::numeric_limits<double>::infinity();
        for(int j=starts[p];j<starts[p+1];++j){int g=geoms[j];double offset[3];mju_sub3(offset,d->geom_xpos+3*g,d->geom_xpos+3*obj);
          double bound=mju_norm3(offset)+m->geom_rbound[g]+m->geom_rbound[obj]+1.;
          double distance=mj_geomDistance(m,d,g,obj,std::min(bound,gap),segment);
          if(distance<gap){gap=distance;a=g;std::copy(segment,segment+6,nearest);}
        }
      }
      if(a<0||!std::isfinite(gap))return 1;
      std::copy(nearest,nearest+3,pa);std::copy(nearest+3,nearest+6,pb);
      mju_sub3(normal,pb,pa);double norm=mju_norm3(normal);if(norm==0.)return 2;mju_scl3(normal,normal,1./norm);
    }
    mj_jac(m,d,ja.data(),nullptr,pa,m->geom_bodyid[a]);mj_jac(m,d,jb.data(),nullptr,pb,m->geom_bodyid[b]);
    for(int k=0;k<m->nv;++k)rows[p*m->nv+k]=normal[0]*(jb[k]-ja[k])+normal[1]*(jb[m->nv+k]-ja[m->nv+k])+normal[2]*(jb[2*m->nv+k]-ja[2*m->nv+k]);
    gaps[p]=gap;
  }
  return 0;
}
