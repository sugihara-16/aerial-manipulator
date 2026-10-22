// Experimental batching of the existing predictor. No changed physical model.
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/serialization/model.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pin = pinocchio;
using Vec=Eigen::VectorXd;
using Mat=Eigen::MatrixXd;
using RowMat=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
struct Drive {int q,v;double kp,kd,limit;};
struct Prediction {
  pin::Model model;
  pin::Data data;
  Vec armature;
  std::vector<Drive> drives;
  std::vector<int> rotors;
  Prediction(const char* binary,const char* metadata):data(model) {
    pin::serialization::loadFromBinary(model,std::string(binary));
    data=pin::Data(model);armature=model.armature;
    auto meta=nlohmann::json::parse(metadata);
    for(const auto& a:meta["actuators"])
      drives.push_back({a["q"],a["v"],a["kp"],a["kd"],a["effort"]});
    for(const auto& r:meta["rotors"])rotors.push_back(r["frame"]);
  }
  int interval(const double* q0,const double* v0,const double* command,
      const double* force,const double* load,double h,int count,bool linearize,double* output) {
    const int nv=model.nv,nq=model.nq,na=drives.size(),nr=rotors.size(),nu=na+nr;
    Vec q=Eigen::Map<const Vec>(q0,nq),v=Eigen::Map<const Vec>(v0,nv);
    const Vec extload=Eigen::Map<const Vec>(load,nv);
    Mat pj=Mat::Zero(nv,nu),vj=pj;
    pin::container::aligned_vector<pin::Force> external(model.njoints,pin::Force::Zero());
    for(int i=0;i<nr;++i){
      const auto& frame=model.frames[rotors[i]];
      external[frame.parentJoint]+=frame.placement.act(pin::Force(Eigen::Vector3d(0,0,force[i]),Eigen::Vector3d::Zero()));
    }
    int last_iterations=0;
    for(int step=0;step<count;++step) {
      std::vector<int> active(na,0),needed(na,0);
      Vec bias0(na),coefficient0(na),coefficient(na),bias(na),torque,acceleration,velocity,raw(na);
      for(int i=0;i<na;++i){const auto& a=drives[i];
        bias0[i]=a.kp*(command[i]-q[a.q]-h*v[a.v])-a.kd*v[a.v];
        coefficient0[i]=a.kp*h*h+a.kd*h;
      }
      bool converged=false;
      for(int iteration=0;iteration<12;++iteration){
        model.armature=armature;torque=extload;
        for(int i=0;i<na;++i){
          coefficient[i]=active[i]?0.:coefficient0[i];
          bias[i]=active[i]?active[i]*drives[i].limit:bias0[i];
          model.armature[drives[i].v]+=coefficient[i];torque[drives[i].v]+=bias[i];
        }
        acceleration=pin::aba(model,data,q,v,torque,external);velocity=v+h*acceleration;
        for(int i=0;i<na;++i){const auto& a=drives[i];
          raw[i]=a.kp*(command[i]-q[a.q]-h*velocity[a.v])-a.kd*velocity[a.v];
          needed[i]=raw[i]>a.limit?1:(raw[i]<-a.limit?-1:0);
        }
        if(active==needed){converged=true;last_iterations=iteration+1;break;}
        active=needed;
      }
      // The Python reference has the exact bounded fallback. Retry the whole
      // original interval there, rather than introducing a second optimizer.
      if(!converged)throw std::runtime_error("active_set_fallback_required");
      Vec applied(na);
      for(int i=0;i<na;++i){
        applied[i]=bias[i]-coefficient[i]*acceleration[drives[i].v];
        double expected=std::max(-drives[i].limit,std::min(drives[i].limit,raw[i]));
        if(std::abs(applied[i]-expected)>1e-10+1e-10*std::abs(expected))
          throw std::runtime_error("drive_prediction_torque_inconsistent");
      }
      if(linearize){
        pin::computeABADerivatives(model,data,q,v,torque,external);
        Mat aq=data.ddq_dq,av=data.ddq_dv,inverse=data.Minv.selfadjointView<Eigen::Upper>();
        Mat input=Mat::Zero(nv,nu);
        for(int i=0;i<na;++i){const auto& a=drives[i];double free=active[i]==0?1.:0.;
          aq.col(a.v)-=inverse.col(a.v)*(a.kp*free);
          av.col(a.v)-=inverse.col(a.v)*((a.kp*h+a.kd)*free);
          input.col(i)=inverse.col(a.v)*(a.kp*free);
        }
        pin::computeJointJacobians(model,data,q);pin::updateFramePlacements(model,data);
        Mat action(nv,nr),jacobian=Mat::Zero(6,nv);
        for(int i=0;i<nr;++i){
          jacobian.setZero();pin::getFrameJacobian(model,data,rotors[i],pin::LOCAL,jacobian);
          action.col(i)=jacobian.row(2).transpose();
        }
        input.rightCols(nr).noalias()=inverse*action;
        Mat derivative=aq*pj+av*vj+input;
        vj+=h*derivative;
        Mat first=Mat::Zero(nv,nv),last=first;Vec delta=h*velocity;
        pin::dIntegrate(model,q,delta,first,pin::ARG0);
        pin::dIntegrate(model,q,delta,last,pin::ARG1);
        // Configuration integration is a product of independent joint
        // groups. Its two Jacobians are block diagonal by joint; multiplying
        // their known zero off-diagonal blocks wastes most of the work.
        for(int joint=1;joint<model.njoints;++joint){
          int k=model.joints[joint].idx_v(),n=model.joints[joint].nv();
          pj.middleRows(k,n)=(first.block(k,k,n,n)*pj.middleRows(k,n)
                           +last.block(k,k,n,n)*(h*vj.middleRows(k,n))).eval();
        }
      }
      q=pin::integrate(model,q,h*velocity);v=velocity;
      Eigen::Map<Vec>(output,nq)=q;output+=nq;
      Eigen::Map<Vec>(output,nv)=v;output+=nv;
      Eigen::Map<Vec>(output,na)=applied;output+=na;
      Eigen::Map<Vec>(output,nv)=acceleration;output+=nv;
      if(linearize){
        Eigen::Map<RowMat>(output,nv,nu)=pj;output+=nv*nu;
        Eigen::Map<RowMat>(output,nv,nu)=vj;output+=nv*nu;
      }
    }
    model.armature=armature;return last_iterations;
  }
};
static thread_local std::string error;
extern "C" {
const char* prediction_error(){return error.c_str();}
void* prediction_new(const char* binary,const char* meta){
  try{return new Prediction(binary,meta);}catch(const std::exception& e){error=e.what();return nullptr;}
}
void prediction_delete(void* handle){delete static_cast<Prediction*>(handle);}
int prediction_interval(void* handle,const double* q,const double* v,const double* command,
    const double* force,const double* load,double step,int count,int linearize,double* output){
  auto* p=static_cast<Prediction*>(handle);
  try{return p->interval(q,v,command,force,load,step,count,linearize,output);}
  catch(const std::exception& e){p->model.armature=p->armature;error=e.what();return -1;}
}
}
