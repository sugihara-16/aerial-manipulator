// Read-only bounds traversal. Actual geom distances remain Python API calls,
// retaining MuJoCo's numerical-error handling. No simulation state is advanced.
#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
extern "C" int mj_collideOBB(const mjtNum*,const mjtNum*,const mjtNum*,const mjtNum*,
                            const mjtNum*,const mjtNum*,mjtNum,mjtNum*,mjtNum*,mjtBool*);
struct Node {int body,geom,index;};
struct Pair {Node a,b;};
static int geom(const mjModel* m,Node n) {
  if(n.geom>=0)return n.geom;
  int k=m->body_bvhadr[n.body]+n.index;
  return m->bvh_child[2*k]<0 && m->bvh_child[2*k+1]<0 ? m->bvh_nodeid[k] : -1;
}
static const double* bound(const mjModel* m,Node n,int g) {
  return g>=0 ? m->geom_aabb+6*g : m->bvh_aabb+6*(m->body_bvhadr[n.body]+n.index);
}
extern "C" int interval_leaves(const mjModel* m,const mjData* d,int count,
                               const int* entities,const double* required,int capacity,int* output) {
  int total=0;std::vector<Pair> pending;pending.reserve(64);
  for(int r=0;r<count;++r){
    const int* e=entities+4*r;pending.push_back({{e[0],e[1],0},{e[2],e[3],0}});
    while(!pending.empty()){
      Pair p=pending.back();pending.pop_back();int ga=geom(m,p.a),gb=geom(m,p.b);
      const double *ba=bound(m,p.a,ga),*bb=bound(m,p.b,gb);
      const double *pa=ga>=0?d->geom_xpos+3*ga:d->xipos+3*p.a.body;
      const double *pb=gb>=0?d->geom_xpos+3*gb:d->xipos+3*p.b.body;
      const double *ra=ga>=0?d->geom_xmat+9*ga:d->ximat+9*p.a.body;
      const double *rb=gb>=0?d->geom_xmat+9*gb:d->ximat+9*p.b.body;
      mjtBool initialize=1;
      if(!mj_collideOBB(ba,bb,pa,ra,pb,rb,required[r]+1e-10,nullptr,nullptr,&initialize))continue;
      if(ga>=0 && gb>=0){
        if(!((m->geom_contype[ga]&m->geom_conaffinity[gb]) ||
             (m->geom_contype[gb]&m->geom_conaffinity[ga])))continue;
        if(total==capacity)return -1;
        output[3*total]=ga;output[3*total+1]=gb;output[3*total+2]=r;++total;continue;
      }
      // Split the non-leaf with the larger bound to limit unnecessary pairs.
      bool split_a=ga<0 && (gb>=0 || ba[3]+ba[4]+ba[5]>=bb[3]+bb[4]+bb[5]);
      Node n=split_a?p.a:p.b;int k=m->body_bvhadr[n.body]+n.index;
      for(int c=0;c<2;++c){int child=m->bvh_child[2*k+c];if(child<0)continue;
        Node next={n.body,-1,child};pending.push_back(split_a?Pair{next,p.b}:Pair{p.a,next});}
    }
  }
  return total;
}

// Exact convex support value in one world direction. Meshes inspect every
// stored vertex; no iterative distance estimate is used as a lower bound.
static double support(const mjModel* m,const mjData* d,int g,const double* n){
  double local[3];mju_mulMatTVec3(local,d->geom_xmat+9*g,n);
  const double* size=m->geom_size+3*g;double value=0.;
  switch(m->geom_type[g]){
    case mjGEOM_BOX:for(int k=0;k<3;++k)value+=std::abs(local[k])*size[k];break;
    case mjGEOM_SPHERE:value=size[0]*mju_norm3(local);break;
    case mjGEOM_ELLIPSOID:for(int k=0;k<3;++k)value+=std::pow(local[k]*size[k],2);value=std::sqrt(value);break;
    case mjGEOM_CYLINDER:value=size[0]*std::hypot(local[0],local[1])+size[1]*std::abs(local[2]);break;
    case mjGEOM_CAPSULE:value=size[0]*mju_norm3(local)+size[1]*std::abs(local[2]);break;
    case mjGEOM_MESH:{
      int id=m->geom_dataid[g],start=m->mesh_vertadr[id],end=start+m->mesh_vertnum[id];
      value=-std::numeric_limits<double>::infinity();
      for(int v=start;v<end;++v){const float* x=m->mesh_vert+3*v;
        value=std::max(value,double(x[0])*local[0]+double(x[1])*local[1]+double(x[2])*local[2]);}break;
    }
    default:return std::numeric_limits<double>::infinity();
  }
  return mju_dot3(d->geom_xpos+3*g,n)+value;
}
extern "C" double interval_separation(const mjModel* m,const mjData* d,int a,int b,const double* witness){
  if(m->geom_type[a]==mjGEOM_PLANE || m->geom_type[b]==mjGEOM_PLANE){
    int p=m->geom_type[a]==mjGEOM_PLANE?a:b,g=p==a?b:a;
    const double* R=d->geom_xmat+9*p;double n[3]={R[2],R[5],R[8]},neg[3];
    mju_scl3(neg,n,-1.);return -support(m,d,g,neg)-mju_dot3(d->geom_xpos+3*p,n)-1e-10;
  }
  double n[3],neg[3];mju_sub3(n,witness+3,witness);
  if(mju_norm3(n)<1e-15)mju_sub3(n,d->geom_xpos+3*b,d->geom_xpos+3*a);
  if(mju_norm3(n)<1e-15)return -std::numeric_limits<double>::infinity();
  mju_normalize3(n);mju_scl3(neg,n,-1.);
  double sa=support(m,d,a,n),sb=support(m,d,b,neg);
  return -sb-sa-1e-10*(1+std::abs(sa)+std::abs(sb));
}
