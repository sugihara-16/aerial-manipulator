#include "sim.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <cmath>

namespace det {
Controller::Controller(Plant& p,const Observation& observation,const Command& held,const Vec& estimate,double tc):
  plant(p),data(p.model),reference_configuration(observation.q),command(held),estimated_thrust(estimate),period(tc){
  if(estimate.size()!=int(p.rotors.size())||!estimate.allFinite())throw std::runtime_error("invalid initial thrust estimate");
  if(observation.q.size()!=p.model.nq||observation.v.size()!=p.model.nv||observation.motor_effort.size()!=int(p.actuators.size())||
    !observation.q.allFinite()||!observation.v.allFinite()||!observation.motor_effort.allFinite())throw std::runtime_error("invalid initial observation dimensions or values");
  if(p.scenario.value("move_duration_s",10.)<=0||p.scenario.value("allocation_gimbal_predictor_s",.05)<=0)
    throw std::runtime_error("positive motion duration and allocation response required");
}
// Low-order next-period mean prediction. Unlike the plant this is a predictor,
// not a measurement of hidden actuator state. Its response constant is declared
// in the diagnostic profile and has not been identified on hardware.
V6 Controller::wrench(const Vec& input,const Vec& q,int base,bool jac,Eigen::MatrixXd* B){
  int nr=plant.rotors.size();V6 result=V6::Zero();if(jac)B->setZero(6,2*nr);
  const auto& root=plant.bases[base];V3 origin=data.oMi[root.joint].translation();
  double tg=plant.scenario.value("allocation_gimbal_predictor_s",0.05);
  double ag=1-tg/period*(1-std::exp(-period/tg));
  double af=plant.thrust_lag>0?1-plant.thrust_lag/period*(1-std::exp(-period/plant.thrust_lag)):1.;
  for(int k=0;k<nr;++k){
    const auto& r=plant.rotors[k];const auto& a=plant.actuators[r.actuator];
    int ancestor=a.joint;while(plant.model.parents[ancestor]!=0)ancestor=plant.model.parents[ancestor];
    if(ancestor!=root.joint)continue;
    const auto& G=data.oMi[a.joint];const auto& F=data.oMf[r.frame];
    // Gimbal axes are authored +x. They cannot be on the assembly reroot path.
    V3 axis=G.rotation().col(0);double angle=ag*(input[nr+k]-q[a.q]);
    Eigen::AngleAxisd rot(angle,axis);
    V3 direction=rot*F.rotation().col(2);
    V3 relative=rot*(F.translation()-G.translation());
    V3 arm=G.translation()+relative-origin;
    double force=(1-af)*estimated_thrust[k]+af*input[k];
    V3 fw=force*direction;
    result.head<3>()+=fw;result.tail<3>()+=arm.cross(fw);
    if(jac){
      B->block<3,1>(0,k)=af*direction;B->block<3,1>(3,k)=arm.cross(af*direction);
      V3 df=ag*force*axis.cross(direction),dp=ag*axis.cross(relative);
      B->block<3,1>(0,nr+k)=df;B->block<3,1>(3,nr+k)=dp.cross(fw)+arm.cross(df);
    }
  }
  return result;
}
Command Controller::update(const Observation& obs,double time){
  if(!obs.valid||!std::isfinite(obs.timestamp)||obs.timestamp>time+1e-10||time-obs.timestamp>plant.scenario.value("max_observation_age_s",0.05))throw std::runtime_error("stale observation; no validated recovery available");
  if(obs.q.size()!=plant.model.nq||obs.v.size()!=plant.model.nv||obs.motor_effort.size()!=int(plant.actuators.size())||!obs.q.allFinite()||!obs.v.allFinite()||!obs.motor_effort.allFinite())throw std::runtime_error("invalid observation; no validated recovery available");
  auto start=Clock::now();
  pinocchio::forwardKinematics(plant.model,data,obs.q,obs.v);
  for(const auto& rotor:plant.rotors)pinocchio::updateFramePlacement(plant.model,data,rotor.frame);
  int nr=plant.rotors.size();Vec inputs(2*nr);
  for(int k=0;k<nr;++k){inputs[k]=command.thrust[k];inputs[nr+k]=command.position[plant.rotors[k].actuator];}
  double duration=plant.scenario.value("move_duration_s",10.);
  double u=std::clamp(time/duration,0.,1.);
  double s=u*u*u*(10+u*(-15+6*u));
  double ds=30*u*u*(1-u)*(1-u)/duration;
  double dds=60*u*(1-u)*(1-2*u)/(duration*duration);
  V3 offset=plant.scenario.contains("translation")?V3(jsonvec(plant.scenario["translation"])):V3::Zero();
  for(size_t bi=0;bi<plant.bases.size();++bi){
    const auto& b=plant.bases[bi];Eigen::Matrix3d R=data.oMi[b.joint].rotation();
    V3 position=reference_configuration.segment<3>(b.q)+s*offset;
    V3 velocity=R*obs.v.segment<3>(b.v), omega=R*obs.v.segment<3>(b.v+3);
    Eigen::Quaterniond qref(reference_configuration[b.q+6],reference_configuration[b.q+3],reference_configuration[b.q+4],reference_configuration[b.q+5]);
    V3 orient=pinocchio::log3(qref.toRotationMatrix()*R.transpose());
    double wn=plant.scenario.value("translation_frequency_rad_s",2.);
    double wa=plant.scenario.value("attitude_frequency_rad_s",3.);
    Vec acceleration=Vec::Zero(plant.model.nv);
    acceleration.segment<3>(b.v)=R.transpose()*(dds*offset+wn*wn*(position-obs.q.segment<3>(b.q))+2*wn*(ds*offset-velocity))
      -obs.v.segment<3>(b.v+3).cross(obs.v.segment<3>(b.v));
    acceleration.segment<3>(b.v+3)=R.transpose()*(wa*wa*orient-2*wa*omega);
    // Root translation acceleration also needs a moment about the root when
    // the center of mass is displaced. The complete inverse-dynamics wrench
    // keeps that cross term; adding m*a only to force would omit it.
    Vec required=pinocchio::rnea(plant.model,data,obs.q,obs.v,acceleration);
    V6 target;target.head<3>()=R*required.segment<3>(b.v);target.tail<3>()=R*required.segment<3>(b.v+3);
    for(int iteration=0;iteration<2;++iteration){
      Eigen::MatrixXd B(6,2*nr);V6 current=wrench(inputs,obs.q,bi,true,&B);
      Vec D=Vec::Ones(2*nr);D.head(nr).setConstant(25.);D.tail(nr).setConstant(0.25);
      Eigen::Matrix<double,6,6> H=B*D.asDiagonal()*B.transpose();H.diagonal().array()+=1e-5;
      V6 delta=H.ldlt().solve(target-current);
      inputs+=D.asDiagonal()*B.transpose()*delta;
      for(int k=0;k<nr;++k){
        const auto& r=plant.rotors[k];const auto& a=plant.actuators[r.actuator];
        inputs[k]=std::clamp(inputs[k],r.lo,r.hi);
        inputs[nr+k]=std::clamp(inputs[nr+k],std::max(a.lo,command.position[r.actuator]-a.speed*period),std::min(a.hi,command.position[r.actuator]+a.speed*period));
      }
      ++plant.counts.allocation;
    }
    double residual=(target-wrench(inputs,obs.q,bi,false)).norm();
    plant.counts.max_alloc_residual=std::max(plant.counts.max_alloc_residual,residual);
  }
  for(int k=0;k<nr;++k){command.thrust[k]=inputs[k];command.position[plant.rotors[k].actuator]=inputs[nr+k];}
  estimated_thrust=command.thrust+(estimated_thrust-command.thrust)*(plant.thrust_lag>0?std::exp(-period/plant.thrust_lag):0.);
  ++plant.counts.control;plant.counts.control_s+=elapsed(start);return command;
}
}
