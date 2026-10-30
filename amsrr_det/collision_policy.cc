// Retain face manifolds for polyhedra and cylinder caps; single witnesses on sides.
// The caller's model is never mutated, including when other models run in threads.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
static mjfCollision original[mjNGEOMTYPES][mjNGEOMTYPES];
static bool installed=false;
static int cylinder_single(const mjModel* m,mjData* d,mjPreContact* c,int g1,int g2,mjtNum margin){
    const int a=std::min(m->geom_type[g1],m->geom_type[g2]);
    const int b=std::max(m->geom_type[g1],m->geom_type[g2]);
    mjModel local=*m;
    local.opt.disableflags|=mjDSBL_MULTICCD;
    int n=original[a][b](&local,d,c,g1,g2,margin);
    if(n!=1)return n;  // Specialized analytic colliders keep their own manifold.
    // The normal cone is closer to the cap than to the radial side only when
    // its axial component dominates. Use the reliable single witness to classify.
    for(int g:{g1,g2})if(m->geom_type[g]==mjGEOM_CYLINDER){
        const mjtNum* R=d->geom_xmat+9*g;
        double axial=c[0].normal[0]*R[2]+c[0].normal[1]*R[5]+c[0].normal[2]*R[8];
        if(axial*axial<0.5)return n;
    }
    mjPreContact candidates[mjMAXCONPAIR];
    local.opt.disableflags&=~mjDSBL_MULTICCD;
    int count=original[a][b](&local,d,candidates,g1,g2,margin);
    if(count<2)return n;
    // A flat cap's candidate normals must agree with the independent witness.
    // Invalid/ill-conditioned manifold output falls back to that witness.
    for(int i=0;i<count;++i){
        if(!std::isfinite(candidates[i].dist) || !(mju_dot3(c[0].normal,candidates[i].normal)>=0.9999))return n;
    }
    std::copy_n(candidates,count,c);
    return count;
}
extern "C" int install_policy(){
    if(mj_version()!=mjVERSION_HEADER)return -1;
    if(installed)return 0;
    for(int a=0;a<mjNGEOMTYPES;++a)for(int b=a;b<mjNGEOMTYPES;++b){
        original[a][b]=mjCOLLISIONFUNC[a][b];
        if((a==mjGEOM_CYLINDER||b==mjGEOM_CYLINDER)&&original[a][b])mjCOLLISIONFUNC[a][b]=cylinder_single;
    }
    installed=true;return 0;
}
