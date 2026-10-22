#pragma once
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <string>
#include <limits>

namespace det {
using Vec = Eigen::VectorXd;
using V3 = Eigen::Vector3d;
using V6 = Eigen::Matrix<double,6,1>;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point a) { return std::chrono::duration<double>(Clock::now()-a).count(); }
struct Actuator { int joint,q,v; double kp,kd,effort,peak,lo,hi,speed,lag; bool gimbal; };
struct Rotor { int frame,actuator,module; double lo,hi; };
struct Base { int joint,q,v; };
struct Patch {
  int frame, target; std::vector<V3> points;
  V3 normal; double offset,kn,cn,kt,ct,mu;
  V3 lower,upper;
  double force_limit,moment_limit;
};
struct State { Vec q,v,f,tau,xi; };
struct Derivative { Vec a,f,tau,xi; };
struct Command { Vec thrust,position,velocity; };
// Sensor-equivalent public interface: no contact forces, friction state, or
// exact rotor thrust can cross this boundary.
struct Observation { double timestamp; Vec q,v,motor_effort; bool valid=true; };
struct Counts {
  long steps=0,rhs=0,control=0,observer=0,allocation=0,contact_nodes=0,accepted_steps=0,rejected_steps=0;
  double physics_s=0,control_s=0,peak_force=0,peak_penetration=0,peak_slip_speed=0;
  double max_joint_speed=0,max_joint_limit_violation=0,max_effort_ratio=0,max_alloc_residual=0;
  double max_speed_ratio=0,peak_patch_force=0,peak_patch_moment=0,max_contact_ratio=0;
};
class Plant {
 public:
  pinocchio::Model model;
  pinocchio::Data data;
  Json meta,scenario;
  std::vector<Actuator> actuators;
  std::vector<Rotor> rotors;
  std::vector<Base> bases;
  std::vector<Patch> patches;
  Counts counts;
  double thrust_lag;
  Plant(const std::string& directory, const Json& scenario);
  State initial() const;
  Command neutralCommand(const State&) const;
  void validateCommand(const Command&) const;
  Derivative rhs(const State&, const Command&, bool count=true, bool prescribed_effort=false);
  void step(State&, const Command&, double h);
  void implicitDriveStep(State&, const Command&, double h);
  void sdirkDriveStep(State&, const Command&, double h);
  State implicitStage(const State&,const Command&,double h);
  double adaptiveDriveStep(State&,const Command&,double maximum_h);
  double suggested_step=0.;
  void updateKinematics(const State&);
  void projectHistory(State&);
  void audit(const State&);
};
class Controller {
 public:
  Plant& plant;
  pinocchio::Data data;
  Vec reference_configuration;
  Command command;
  Vec estimated_thrust;
  double period;
  Controller(Plant&, const Observation&, const Command&, const Vec& initial_thrust_estimate, double);
  Command update(const Observation&, double);
 private:
  V6 wrench(const Vec&, const Vec&, int, bool jacobian, Eigen::MatrixXd* B=nullptr);
};
Json run(Plant&,double,double,bool,const std::string&);
Vec jsonvec(const Json& j);
Json vecjson(const Vec& v);
}
