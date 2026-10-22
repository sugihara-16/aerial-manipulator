#include "sim.hpp"
#include <pinocchio/serialization/model.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <fstream>
#include <cmath>

namespace det {
Vec jsonvec(const Json& j) { Vec v(j.size()); for(int i=0;i<v.size();++i)v[i]=j[i].get<double>(); return v; }
Json vecjson(const Vec& v) { Json j=Json::array(); for(double x:v)j.push_back(x);return j; }
Plant::Plant(const std::string& dir,const Json& sc): data(model),scenario(sc) {
  pinocchio::serialization::loadFromBinary(model,dir+"/model.bin");
  data=pinocchio::Data(model);
  std::ifstream in(dir+"/model.json");in>>meta;
  if(meta.at("schema")!="amsrr_det.model.v1")throw std::runtime_error("model schema");
  thrust_lag=sc.value("thrust_lag_s",0.03);
  if(thrust_lag<0)throw std::runtime_error("invalid thrust lag");
  std::string method=sc.value("integrator",std::string("explicit_midpoint"));
  if(method!="explicit_midpoint"&&method!="drive_implicit_euler"&&method!="drive_sdirk2"&&method!="adaptive_implicit_euler")throw std::runtime_error("unknown integrator");
  if(method=="adaptive_implicit_euler"){
    for(const std::string key:{"position_m","angle_rad","linear_velocity_m_s","angular_velocity_rad_s","joint_velocity_rad_s","thrust_n","torque_nm","history_m"})
      if(!sc.contains("step_tolerances")||!sc["step_tolerances"].contains(key)||sc["step_tolerances"][key].get<double>()<=0||!std::isfinite(sc["step_tolerances"][key].get<double>()))
        throw std::runtime_error("adaptive integration requires explicit positive step_tolerances");
  }
  for(const auto& a:meta["actuators"]){
    bool g=a["role"]=="vectoring";
    double lag=sc.value(g?"gimbal_torque_lag_s":"dock_torque_lag_s",0.);
    if(lag<0)throw std::runtime_error("invalid torque lag");
    actuators.push_back({a["joint"],a["q"],a["v"],a["kp"],a["kd"],a["effort"],a["peak"],a["lower"],a["upper"],a["speed"],lag,g});
  }
  for(const auto& r:meta["rotors"]) rotors.push_back({r["frame"],r["gimbal"],r["module"],r["minimum"],r["maximum"]});
  for(const auto& b:meta["bases"]) bases.push_back({b["joint"],b["q"],b["v"]});
  for(const auto& p:sc.value("patches",Json::array())){
    Patch z; z.frame=meta["frames"].at(p.at("frame").get<std::string>()).get<int>();
    z.target=p.contains("target_frame")?meta["frames"].at(p.at("target_frame").get<std::string>()).get<int>():-1;
    for(const auto& x:p.at("points"))z.points.push_back(jsonvec(x));
    if(z.points.size()!=4)throw std::runtime_error("each patch requires four points");
    z.normal=jsonvec(p.at("normal"));
    if(std::abs(z.normal.norm()-1)>1e-10)throw std::runtime_error("contact normal not unit");
    z.offset=p.at("offset");z.kn=p.at("kn").get<double>()/4.;z.cn=p.at("cn").get<double>()/4.;
    z.kt=p.at("kt").get<double>()/4.;z.ct=p.at("ct").get<double>()/4.;z.mu=p.at("mu");
    z.lower=jsonvec(p.at("lower"));z.upper=jsonvec(p.at("upper"));
    z.force_limit=p.value("max_force_n",std::numeric_limits<double>::infinity());
    z.moment_limit=p.value("max_moment_nm",std::numeric_limits<double>::infinity());
    if(z.force_limit<=0||z.moment_limit<=0)throw std::runtime_error("invalid contact load limit");
    if(z.kn<=0||z.cn<0||z.kt<=0||z.ct<0||z.mu<0)throw std::runtime_error("invalid contact material");
    patches.push_back(z);
  }
  if(method=="drive_sdirk2"&&!patches.empty())
    throw std::runtime_error("experimental SDIRK2 currently excludes contact/history; use verified contact integrator");
}
State Plant::initial() const {
  State s{pinocchio::neutral(model),Vec::Zero(model.nv),Vec::Zero(rotors.size()),Vec::Zero(actuators.size()),Vec::Zero(12*patches.size())};
  for(size_t i=0;i<bases.size();++i){
    s.q[bases[i].q+2]=1.;
    if(scenario.contains("base_positions")) s.q.segment<3>(bases[i].q)=jsonvec(scenario["base_positions"][i]);
  }
  if(scenario.contains("q0"))s.q=jsonvec(scenario["q0"]);
  if(scenario.contains("v0"))s.v=jsonvec(scenario["v0"]);
  if(scenario.contains("f0"))s.f=jsonvec(scenario["f0"]);
  if(scenario.contains("tau0"))s.tau=jsonvec(scenario["tau0"]);
  if(scenario.contains("xi0"))s.xi=jsonvec(scenario["xi0"]);
  if(s.q.size()!=model.nq||s.v.size()!=model.nv||s.f.size()!=int(rotors.size())||s.tau.size()!=int(actuators.size())||s.xi.size()!=int(12*patches.size()))throw std::runtime_error("initial state dimension");
  if(!pinocchio::isNormalized(model,s.q,1e-8))throw std::runtime_error("invalid initial quaternion");
  if(!s.q.allFinite()||!s.v.allFinite()||!s.f.allFinite()||!s.tau.allFinite()||!s.xi.allFinite())throw std::runtime_error("nonfinite initial state");
  for(size_t i=0;i<rotors.size();++i)if(s.f[i]<rotors[i].lo||s.f[i]>rotors[i].hi)throw std::runtime_error("initial thrust outside limits");
  for(size_t i=0;i<actuators.size();++i){const auto& a=actuators[i];
    if(s.q[a.q]<a.lo||s.q[a.q]>a.hi||std::abs(s.v[a.v])>a.speed||std::abs(s.tau[i])>a.effort)
      throw std::runtime_error("initial joint state outside limits");
  }
  return s;
}
Command Plant::neutralCommand(const State& s) const {
  Command u{Vec::Zero(rotors.size()),Vec::Zero(actuators.size()),Vec::Zero(actuators.size())};
  for(size_t i=0;i<actuators.size();++i)u.position[i]=s.q[actuators[i].q];
  if(scenario.contains("command")){
    const auto& j=scenario["command"];
    if(j.contains("thrust"))u.thrust=jsonvec(j["thrust"]);
    if(j.contains("position"))u.position=jsonvec(j["position"]);
    if(j.contains("velocity"))u.velocity=jsonvec(j["velocity"]);
  }
  validateCommand(u);return u;
}
void Plant::validateCommand(const Command& u) const {
  if(u.thrust.size()!=int(rotors.size())||u.position.size()!=int(actuators.size())||u.velocity.size()!=int(actuators.size()))throw std::runtime_error("command dimension");
  if(!u.thrust.allFinite()||!u.position.allFinite()||!u.velocity.allFinite())throw std::runtime_error("nonfinite command");
  for(int i=0;i<u.thrust.size();++i)if(u.thrust[i]<rotors[i].lo||u.thrust[i]>rotors[i].hi)throw std::runtime_error("thrust command limit");
  for(int i=0;i<u.position.size();++i)if(u.position[i]<actuators[i].lo||u.position[i]>actuators[i].hi||std::abs(u.velocity[i])>actuators[i].speed)throw std::runtime_error("joint command limit");
}
void Plant::updateKinematics(const State& s){
  pinocchio::forwardKinematics(model,data,s.q,s.v);
  // Only the contact frames need world placements. Rotor forces are applied
  // directly in their parent-joint frame below; unqueried mesh frames are not
  // a hidden part of the dynamics and need not be recomputed here.
  for(const auto& p:patches){pinocchio::updateFramePlacement(model,data,p.frame);if(p.target>=0)pinocchio::updateFramePlacement(model,data,p.target);}
}
static V3 pointVelocity(const pinocchio::Model& m,const pinocchio::Data& d,int fid,const V3& local){
  const auto& fr=m.frames[fid];V3 inJoint=fr.placement.act(local);
  return d.oMi[fr.parentJoint].rotation()*(d.v[fr.parentJoint].linear()+d.v[fr.parentJoint].angular().cross(inJoint));
}
static void addForce(const pinocchio::Model& m,const pinocchio::Data& d,int fid,const V3& worldPoint,const V3& worldForce,pinocchio::container::aligned_vector<pinocchio::Force>& ext){
  int j=m.frames[fid].parentJoint;
  V3 force=d.oMi[j].rotation().transpose()*worldForce;
  V3 point=d.oMi[j].actInv(worldPoint);
  ext[j]+=pinocchio::Force(force,point.cross(force));
}
Derivative Plant::rhs(const State& s,const Command& u,bool count,bool prescribed_effort){
  auto t=Clock::now(); if(!patches.empty())updateKinematics(s);
  pinocchio::container::aligned_vector<pinocchio::Force> ext(model.njoints,pinocchio::Force::Zero());
  Vec effort=Vec::Zero(model.nv);
  Derivative out{Vec::Zero(model.nv),Vec::Zero(s.f.size()),Vec::Zero(s.tau.size()),Vec::Zero(s.xi.size())};
  if(thrust_lag>0)out.f=(u.thrust-s.f)/thrust_lag;
  for(size_t i=0;i<actuators.size();++i){
    const auto& a=actuators[i];
    double desired=a.kp*(u.position[i]-s.q[a.q])+a.kd*(u.velocity[i]-s.v[a.v]);
    desired=std::clamp(desired,-a.effort,a.effort);
    if(a.lag>0)out.tau[i]=(desired-s.tau[i])/a.lag;
    effort[a.v]=(a.lag>0||prescribed_effort)?s.tau[i]:desired;
  }
  for(size_t i=0;i<rotors.size();++i){
    const auto& frame=model.frames[rotors[i].frame];
    ext[frame.parentJoint]+=frame.placement.act(pinocchio::Force(V3(0.,0.,thrust_lag>0?s.f[i]:u.thrust[i]),V3::Zero()));
  }
  int node=0;
  for(const auto& p:patches){
    pinocchio::SE3 target=p.target<0?pinocchio::SE3::Identity():data.oMf[p.target];
    V3 patch_force=V3::Zero(),patch_moment=V3::Zero();
    for(const V3& point:p.points){
      V3 world=data.oMf[p.frame].act(point),relative=target.actInv(world);
      double gap=p.normal.dot(relative)-p.offset;
      bool within=(relative.array()>=p.lower.array()).all()&&(relative.array()<=p.upper.array()).all();
      if(gap<0&&within){
        V3 vt=pointVelocity(model,data,p.frame,point);
        if(p.target>=0)vt-=pointVelocity(model,data,p.target,relative);
        vt=target.rotation().transpose()*vt;
        double vn=p.normal.dot(vt);V3 tangent=vt-p.normal*vn;
        double fn=std::max(0.,-p.kn*gap-p.cn*vn);
        V3 xi=s.xi.segment<3>(3*node);xi-=p.normal*p.normal.dot(xi);
        V3 ft=-p.kt*xi-p.ct*tangent;double cap=p.mu*fn;
        if(ft.norm()>cap&&ft.norm()>0)ft*=cap/ft.norm();
        V3 force=target.rotation()*(p.normal*fn+ft);
        patch_force+=force;patch_moment+=(world-data.oMf[p.frame].translation()).cross(force);
        addForce(model,data,p.frame,world,force,ext);
        if(p.target>=0)addForce(model,data,p.target,world,-force,ext);
        out.xi.segment<3>(3*node)=tangent;
        if(count){counts.peak_force=std::max(counts.peak_force,force.norm());counts.peak_penetration=std::max(counts.peak_penetration,-gap);counts.peak_slip_speed=std::max(counts.peak_slip_speed,tangent.norm());}
      }
      ++node;
    }
    if(count){
      counts.peak_patch_force=std::max(counts.peak_patch_force,patch_force.norm());
      counts.peak_patch_moment=std::max(counts.peak_patch_moment,patch_moment.norm());
      counts.max_contact_ratio=std::max({counts.max_contact_ratio,patch_force.norm()/p.force_limit,patch_moment.norm()/p.moment_limit});
    }
  }
  out.a=pinocchio::aba(model,data,s.q,s.v,effort,ext);
  if(count){++counts.rhs;counts.contact_nodes+=node;counts.physics_s+=elapsed(t);}
  return out;
}
void Plant::projectHistory(State& s){
  if(patches.empty())return;
  updateKinematics(s);int node=0;
  for(const auto& p:patches){
    pinocchio::SE3 T=p.target<0?pinocchio::SE3::Identity():data.oMf[p.target];
    for(const auto& point:p.points){
      V3 world=data.oMf[p.frame].act(point),rel=T.actInv(world);
      double gap=p.normal.dot(rel)-p.offset;
      bool within=(rel.array()>=p.lower.array()).all()&&(rel.array()<=p.upper.array()).all();
      V3 xi=s.xi.segment<3>(3*node);
      if(gap>=0||!within)xi.setZero();
      else{
        V3 vel=pointVelocity(model,data,p.frame,point);
        if(p.target>=0)vel-=pointVelocity(model,data,p.target,rel);
        vel=T.rotation().transpose()*vel;
        double vn=p.normal.dot(vel);V3 vt=vel-vn*p.normal;
        xi-=p.normal*p.normal.dot(xi);
        double fn=std::max(0.,-p.kn*gap-p.cn*vn);
        V3 trial=-p.kt*xi-p.ct*vt;
        if(trial.norm()>p.mu*fn&&trial.norm()>0){trial*=p.mu*fn/trial.norm();xi=-(trial+p.ct*vt)/p.kt;}
      }
      s.xi.segment<3>(3*node)=xi;++node;
    }
  }
}
void Plant::step(State& s,const Command& u,double h){
  if(scenario.value("integrator",std::string("explicit_midpoint"))=="drive_sdirk2"){
    sdirkDriveStep(s,u,h);return;
  }
  if(scenario.value("integrator",std::string("explicit_midpoint"))=="drive_implicit_euler"){
    implicitDriveStep(s,u,h);return;
  }
  // Explicit midpoint on the joint manifold; control and observations are held.
  Derivative a=rhs(s,u);State mid=s;
  mid.q=pinocchio::integrate(model,s.q,s.v*(h*.5));mid.v+=h*.5*a.a;
  mid.f+=h*.5*a.f;mid.tau+=h*.5*a.tau;mid.xi+=h*.5*a.xi;
  Derivative b=rhs(mid,u);
  s.q=pinocchio::integrate(model,s.q,mid.v*h);s.v+=h*b.a;
  s.f+=h*b.f;s.tau+=h*b.tau;s.xi+=h*b.xi;
  if(thrust_lag==0)s.f=u.thrust;
  for(size_t i=0;i<actuators.size();++i){const auto& a=actuators[i];if(a.lag==0)s.tau[i]=std::clamp(a.kp*(u.position[i]-s.q[a.q])+a.kd*(u.velocity[i]-s.v[a.v]),-a.effort,a.effort);}
  projectHistory(s);++counts.steps;audit(s);
}
void Plant::implicitDriveStep(State& s,const Command& u,double h){
  // Semi-implicit Euler for configuration, backward Euler for the torque/PD
  // subsystem. Eliminating each scalar servo adds a nonnegative diagonal to
  // the articulated inertia, so the same O(N) ABA can solve the coupled step.
  // Saturated drives form an active set; no physical limit is increased.
  Vec original_armature=model.armature,coefficient=Vec::Zero(actuators.size()),bias=coefficient;
  std::vector<int> active(actuators.size(),0);Derivative d;
  State working=s;
  const double decay=thrust_lag>0?std::exp(-h/thrust_lag):0.;
  working.f=u.thrust+(s.f-u.thrust)*(thrust_lag/h)*(1-decay);
  bool solved=false;
  try{
    for(int iteration=0;iteration<8;++iteration){
      model.armature=original_armature;
      for(size_t i=0;i<actuators.size();++i){
        const auto& a=actuators[i];double beta=h/(h+a.lag);
        if(active[i]){bias[i]=(1-beta)*s.tau[i]+beta*active[i]*a.effort;coefficient[i]=0.;}
        else{
          bias[i]=(1-beta)*s.tau[i]+beta*(a.kp*(u.position[i]-s.q[a.q]-h*s.v[a.v])+a.kd*(u.velocity[i]-s.v[a.v]));
          coefficient[i]=beta*(a.kp*h*h+a.kd*h);
        }
        model.armature[a.v]+=coefficient[i];working.tau[i]=bias[i];
      }
      // The servo has already been eliminated into bias and added inertia.
      d=rhs(working,u,true,true);
      bool consistent=true;
      for(size_t i=0;i<actuators.size();++i){
        const auto& a=actuators[i];double vn=s.v[a.v]+h*d.a[a.v];
        double raw=a.kp*(u.position[i]-s.q[a.q]-h*vn)+a.kd*(u.velocity[i]-vn);
        int needed=raw>a.effort?1:(raw<-a.effort?-1:0);
        if(needed!=active[i]){consistent=false;active[i]=needed;}
      }
      if(consistent){solved=true;break;}
    }
  }catch(...){model.armature=original_armature;throw;}
  model.armature=original_armature;
  if(!solved)throw std::runtime_error("implicit drive active set did not converge");
  for(size_t i=0;i<actuators.size();++i)s.tau[i]=bias[i]-coefficient[i]*d.a[actuators[i].v];
  s.v+=h*d.a;s.q=pinocchio::integrate(model,s.q,s.v*h);
  s.f=u.thrust+(s.f-u.thrust)*decay;s.xi+=h*d.xi;
  projectHistory(s);++counts.steps;audit(s);
}
State Plant::implicitStage(const State& base,const Command& u,double h){
  // Solve a DIRK stage, including body geometry/velocity at the stage state.
  // Servo elimination is exact for revolute joint coordinates; other nonlinear
  // forces are iterated, not replaced by artificial damping or frozen bodies.
  State stage=base;stage.q=pinocchio::integrate(model,base.q,h*base.v);
  stage.f=thrust_lag>0?(base.f+(h/thrust_lag)*u.thrust)/(1+h/thrust_lag):u.thrust;
  Vec original=model.armature,bias=Vec::Zero(actuators.size()),coefficient=bias;
  std::vector<int> active(actuators.size(),0);
  bool converged=false;
  try{
    for(int iteration=0;iteration<16;++iteration){
      model.armature=original;State working=stage;
      for(size_t i=0;i<actuators.size();++i){
        const auto& a=actuators[i];double beta=h/(h+a.lag);
        if(active[i]){bias[i]=(1-beta)*base.tau[i]+beta*active[i]*a.effort;coefficient[i]=0.;}
        else{
          bias[i]=(1-beta)*base.tau[i]+beta*(a.kp*(u.position[i]-base.q[a.q]-h*base.v[a.v])+a.kd*(u.velocity[i]-base.v[a.v]));
          coefficient[i]=beta*(a.kp*h*h+a.kd*h);
        }
        model.armature[a.v]+=coefficient[i];working.tau[i]=bias[i];
      }
      Derivative derivative=rhs(working,u,true,true);
      State next=stage;next.v=base.v+h*derivative.a;
      next.q=pinocchio::integrate(model,base.q,h*next.v);
      bool consistent=true;
      for(size_t i=0;i<actuators.size();++i){
        const auto& a=actuators[i];
        next.tau[i]=bias[i]-coefficient[i]*derivative.a[a.v];
        double raw=a.kp*(u.position[i]-next.q[a.q])+a.kd*(u.velocity[i]-next.v[a.v]);
        int needed=raw>a.effort?1:(raw<-a.effort?-1:0);
        if(needed!=active[i]){consistent=false;active[i]=needed;}
      }
      double change=(next.v-stage.v).lpNorm<Eigen::Infinity>();
      stage=next;
      if(consistent&&change<1e-10){converged=true;break;}
    }
  }catch(...){model.armature=original;throw;}
  model.armature=original;
  if(!converged)throw std::runtime_error("SDIRK nonlinear stage failed to converge");
  return stage;
}
void Plant::sdirkDriveStep(State& s,const Command& u,double h){
  // Alexander two-stage SDIRK, gamma = 1 - 1/sqrt(2). The composition of
  // free-flyer retractions is second order locally; no claim at contact events.
  constexpr double gamma=1.-0.7071067811865475244;
  State first=implicitStage(s,u,gamma*h);State predictor=s;
  predictor.q=pinocchio::integrate(model,s.q,h*(1-gamma)*first.v);
  predictor.v+=(1-gamma)/gamma*(first.v-s.v);
  predictor.f+=(1-gamma)/gamma*(first.f-s.f);
  for(size_t i=0;i<actuators.size();++i)
    if(actuators[i].lag>0)predictor.tau[i]+=(1-gamma)/gamma*(first.tau[i]-s.tau[i]);
  State second=implicitStage(predictor,u,gamma*h);
  // Internal stage extrema are samples, not a continuous-time peak guarantee.
  audit(first);audit(second);s=second;++counts.steps;
}
void Plant::audit(const State& s){
  if(!s.q.allFinite()||!s.v.allFinite()||!s.f.allFinite()||!s.tau.allFinite()||!s.xi.allFinite())throw std::runtime_error("nonfinite physics");
  for(size_t i=0;i<actuators.size();++i){const auto& a=actuators[i];
    counts.max_joint_speed=std::max(counts.max_joint_speed,std::abs(s.v[a.v]));
    counts.max_speed_ratio=std::max(counts.max_speed_ratio,std::abs(s.v[a.v])/a.speed);
    counts.max_joint_limit_violation=std::max({counts.max_joint_limit_violation,a.lo-s.q[a.q],s.q[a.q]-a.hi});
    counts.max_effort_ratio=std::max(counts.max_effort_ratio,std::abs(s.tau[i])/a.effort);
  }
  if(s.v.norm()>10000||s.q.head<3>().norm()>100000)throw std::runtime_error("divergent physics");
}
double Plant::adaptiveDriveStep(State& s,const Command& u,double maximum_h){
  const auto& tol=scenario.at("step_tolerances");
  double h=suggested_step>0?std::min(maximum_h,suggested_step):maximum_h;
  const double floor=scenario.value("minimum_step_s",1e-7);
  if(floor<=0||!std::isfinite(floor))throw std::runtime_error("invalid minimum integration step");
  for(int trial=0;trial<20;++trial){
    Counts before=counts;State coarse=s,fine=s;bool evaluated=true;
    try{
      implicitDriveStep(coarse,u,h);
      // Trial-step extrema cannot enter accepted-trajectory peak metrics.
      counts.peak_force=before.peak_force;counts.peak_penetration=before.peak_penetration;
      counts.peak_slip_speed=before.peak_slip_speed;counts.max_joint_speed=before.max_joint_speed;
      counts.max_joint_limit_violation=before.max_joint_limit_violation;counts.max_effort_ratio=before.max_effort_ratio;
      counts.max_speed_ratio=before.max_speed_ratio;counts.peak_patch_force=before.peak_patch_force;
      counts.peak_patch_moment=before.peak_patch_moment;counts.max_contact_ratio=before.max_contact_ratio;
      implicitDriveStep(fine,u,h*.5);implicitDriveStep(fine,u,h*.5);
    }catch(const std::exception&){evaluated=false;}
    double error=1e100;
    if(evaluated){
      Vec dq=pinocchio::difference(model,coarse.q,fine.q),dv=coarse.v-fine.v;
      Vec qscale=Vec::Constant(model.nv,tol["angle_rad"].get<double>());
      Vec vscale=Vec::Constant(model.nv,tol["joint_velocity_rad_s"].get<double>());
      auto set_base=[&](int iv){qscale.segment<3>(iv).setConstant(tol["position_m"]);vscale.segment<3>(iv).setConstant(tol["linear_velocity_m_s"]);vscale.segment<3>(iv+3).setConstant(tol["angular_velocity_rad_s"]);};
      for(const auto& b:bases)set_base(b.v);
      if(!meta["object"].is_null())set_base(meta["object"]["v"]);
      error=std::max((dq.cwiseQuotient(qscale)).lpNorm<Eigen::Infinity>(),(dv.cwiseQuotient(vscale)).lpNorm<Eigen::Infinity>());
      if(s.f.size())error=std::max(error,(fine.f-coarse.f).lpNorm<Eigen::Infinity>()/tol["thrust_n"].get<double>());
      if(s.tau.size())error=std::max(error,(fine.tau-coarse.tau).lpNorm<Eigen::Infinity>()/tol["torque_nm"].get<double>());
      if(s.xi.size())error=std::max(error,(fine.xi-coarse.xi).lpNorm<Eigen::Infinity>()/tol["history_m"].get<double>());
    }
    if(evaluated&&std::isfinite(error)&&error<=1.){
      s=fine;++counts.accepted_steps;
      suggested_step=h*std::clamp(.8/std::sqrt(std::max(error,1e-12)),.5,2.);
      return h;
    }
    counts.peak_force=before.peak_force;counts.peak_penetration=before.peak_penetration;
    counts.peak_slip_speed=before.peak_slip_speed;counts.max_joint_speed=before.max_joint_speed;
    counts.max_joint_limit_violation=before.max_joint_limit_violation;counts.max_effort_ratio=before.max_effort_ratio;
    counts.max_speed_ratio=before.max_speed_ratio;counts.peak_patch_force=before.peak_patch_force;
    counts.peak_patch_moment=before.peak_patch_moment;counts.max_contact_ratio=before.max_contact_ratio;
    ++counts.rejected_steps;
    h*=std::isfinite(error)?std::clamp(.8/std::sqrt(std::max(error,1.)),.1,.5):.1;
    if(h<floor)throw std::runtime_error("adaptive integration minimum step exhausted");
  }
  throw std::runtime_error("adaptive integration trial budget exhausted");
}
}
