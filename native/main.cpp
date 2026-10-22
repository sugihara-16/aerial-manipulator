#include "sim.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <fstream>
#include <iostream>
#include <deque>
#include <iomanip>

namespace det {
Json run(Plant& p,double h,double tc,bool closed,const std::string& trajectory){
  double horizon=p.scenario.at("duration_s"),delay=p.scenario.value("observation_delay_s",0.);
  if(!std::isfinite(h)||!std::isfinite(tc)||!std::isfinite(horizon)||!std::isfinite(delay)||h<=0||tc<=0||horizon<=0||delay<0||h>tc)throw std::runtime_error("invalid time profile");
  State state=p.initial();Command command=p.neutralCommand(state);
  if(closed&&p.thrust_lag>0&&!p.scenario.contains("initial_thrust_estimate"))
    throw std::runtime_error("initial thrust estimate/history required; hidden plant state is not an observation");
  Vec thrust_estimate=p.scenario.contains("initial_thrust_estimate")?jsonvec(p.scenario["initial_thrust_estimate"]):Vec::Zero(p.rotors.size());
  Observation observed{0.,state.q,state.v,state.tau,true};
  if(p.scenario.contains("initial_observation")){
    const auto& packet=p.scenario["initial_observation"];
    observed={packet.at("timestamp"),jsonvec(packet.at("q")),jsonvec(packet.at("v")),jsonvec(packet.at("motor_effort")),packet.value("valid",true)};
    if(observed.timestamp> -delay+1e-12)throw std::runtime_error("initial observation violates declared delivery delay");
  }else if(closed&&delay>0)throw std::runtime_error("delayed observation requires initial observation history");
  Controller controller(p,observed,command,thrust_estimate,tc);
  if(!pinocchio::isNormalized(p.model,observed.q,1e-8))throw std::runtime_error("invalid observation quaternion");
  Json schedule=p.scenario.value("command_schedule",Json::array());size_t next_command=0;
  double last_time=-1.;
  if(closed&&!schedule.empty())throw std::runtime_error("cannot combine command replay and closed-loop control");
  for(const auto& row:schedule){
    double t=row.at("time");
    if(t<0||t<=last_time||std::abs(t/tc-std::round(t/tc))>1e-7)throw std::runtime_error("command times must be ordered control ticks");
    p.validateCommand({jsonvec(row.at("thrust")),jsonvec(row.at("position")),jsonvec(row.at("velocity"))});last_time=t;
  }
  if(!schedule.empty()&&schedule.front().at("time").get<double>()!=0.)throw std::runtime_error("replay must begin at time zero");
  std::deque<Observation> queue;
  std::ofstream trace;if(!trajectory.empty()){trace.open(trajectory);trace<<std::setprecision(17);}
  Json samples=Json::array();double time=0;long cycle=0;bool finite=true;std::string reason="horizon_reached";
  double max_position_error=0.,max_velocity=0.,max_attitude_error=0.;auto start=Clock::now();
  try{
    while(time<horizon-1e-12){
      double tick=cycle*tc;
      if(time>=tick-1e-12){
        queue.push_back({tick,state.q,state.v,state.tau,true});
        while(!queue.empty()&&queue.front().timestamp<=time-delay+1e-12){observed=queue.front();queue.pop_front();}
        if(time>=p.scenario.value("observation_dropout_at_s",1e100))observed.valid=false;
        ++p.counts.observer;
        if(closed)command=controller.update(observed,time);
        else{
          while(next_command<schedule.size()&&schedule[next_command]["time"].get<double>()<=time+1e-12){
            const auto& row=schedule[next_command++];
            command={jsonvec(row.at("thrust")),jsonvec(row.at("position")),jsonvec(row.at("velocity"))};
          }
          ++p.counts.control; // actual replay/hold dispatch, not a fictitious solve
        }
        Json row={{"time",time},{"q",vecjson(state.q)},{"v",vecjson(state.v)},{"thrust_command",vecjson(command.thrust)},
          {"position_command",vecjson(command.position)},{"velocity_command",vecjson(command.velocity)}};
        if(trace)trace<<row.dump()<<'\n';
        if(cycle%std::max(1L,long(std::llround(1./tc)))==0)samples.push_back(row);
        ++cycle;
      }
      double boundary=std::min(horizon,cycle*tc);
      double dt=std::min(h,boundary-time);
      if(boundary-time<=h+1e-10)dt=boundary-time;
      if(dt<=1e-14)throw std::runtime_error("time scheduler stalled");
      if(p.scenario.value("integrator",std::string())=="adaptive_implicit_euler")dt=p.adaptiveDriveStep(state,command,dt);
      else {p.step(state,command,dt);++p.counts.accepted_steps;}
      time+=dt;
      if(std::abs(time-boundary)<1e-10)time=boundary;
      max_velocity=std::max(max_velocity,state.v.norm());
      for(const auto& b:p.bases){
        double u=std::clamp(time/p.scenario.value("move_duration_s",10.),0.,1.);
        double smooth=u*u*u*(10+u*(-15+6*u));
        V3 target=controller.reference_configuration.segment<3>(b.q);
        if(p.scenario.contains("translation"))target+=smooth*V3(jsonvec(p.scenario["translation"]));
        max_position_error=std::max(max_position_error,(state.q.segment<3>(b.q)-target).norm());
        Eigen::Quaterniond a(state.q[b.q+6],state.q[b.q+3],state.q[b.q+4],state.q[b.q+5]);
        const auto& qi=controller.reference_configuration;
        Eigen::Quaterniond ref(qi[b.q+6],qi[b.q+3],qi[b.q+4],qi[b.q+5]);
        max_attitude_error=std::max(max_attitude_error,a.angularDistance(ref));
      }
      if(p.scenario.value("stop_on_limit_violation",true)&&
        (p.counts.max_joint_limit_violation>1e-6||p.counts.max_speed_ratio>1.000001||p.counts.max_effort_ratio>1.00001||p.counts.max_contact_ratio>1.000001)){
        reason="physical_limit_violation";break;
      }
      if(p.counts.peak_penetration>p.scenario.value("max_penetration_m",0.002)){reason="penetration_limit";break;}
    }
  }catch(const std::exception& e){finite=false;reason=e.what();}
  auto& c=p.counts;
  Json result={{"schema","amsrr_det.rollout.v1"},{"horizon_s",horizon},{"simulated_s",time},{"wall_s",elapsed(start)},
    {"step_s",h},{"control_period_s",tc},{"closed_loop",closed},{"stop_reason",reason},{"evaluation_without_exception",finite},
    {"finite",state.q.allFinite()&&state.v.allFinite()&&state.f.allFinite()&&state.tau.allFinite()&&state.xi.allFinite()},
    {"full_horizon",time>=horizon-1e-10},{"task_accepted",false},
    {"counts",{{"steps",c.steps},{"accepted_steps",c.accepted_steps},{"rejected_steps",c.rejected_steps},{"rhs",c.rhs},{"control",c.control},{"observer",c.observer},{"allocation_factorizations",c.allocation},{"contact_nodes",c.contact_nodes}}},
    {"timing",{{"rhs_s",c.physics_s},{"controller_s",c.control_s}}},
    {"metrics",{{"max_position_error_m",max_position_error},{"max_attitude_error_rad",max_attitude_error},{"max_velocity_norm",max_velocity},
      {"peak_node_force_n",c.peak_force},{"peak_penetration_m",c.peak_penetration},{"peak_slip_speed_m_s",c.peak_slip_speed},
      {"peak_patch_force_n",c.peak_patch_force},{"peak_patch_moment_nm",c.peak_patch_moment},{"max_contact_load_ratio",c.max_contact_ratio},{"max_joint_speed_ratio",c.max_speed_ratio},
      {"max_joint_speed_rad_s",c.max_joint_speed},{"max_joint_limit_violation_rad",c.max_joint_limit_violation},
      {"max_effort_ratio",c.max_effort_ratio},{"max_allocation_residual",c.max_alloc_residual}}},
    {"final",{{"q",vecjson(state.q)},{"v",vecjson(state.v)},{"f",vecjson(state.f)},{"tau",vecjson(state.tau)},{"xi",vecjson(state.xi)}}},
    {"samples",samples},{"not_implemented",{"whole-shape collision validation","connector load limits","assembly and latch transitions","task completion and recovery","object error feedback","load observability and handover"}}};
  return result;
}
}
int main(int argc,char**argv){
  if(argc<5){std::cerr<<"usage: det_rollout MODEL_DIR SCENARIO_JSON STEP_S OUTPUT_JSON [closed] [trace.jsonl]\n";return 2;}
  try{
    auto start=det::Clock::now();std::ifstream stream(argv[2]);det::Json scenario;stream>>scenario;
    det::Plant plant(argv[1],scenario);double prep=det::elapsed(start);
    det::Json result;
    if(scenario.value("probe",false)){
      auto state=plant.initial();auto derivative=plant.rhs(state,plant.neutralCommand(state));
      result={{"acceleration",det::vecjson(derivative.a)},{"thrust_rate",det::vecjson(derivative.f)},
        {"effort_rate",det::vecjson(derivative.tau)},{"history_rate",det::vecjson(derivative.xi)},
        {"wall_s",0.},{"simulated_s",0.},{"stop_reason","probe"},{"finite",true}};
    }else result=det::run(plant,std::stod(argv[3]),scenario.value("control_period_s",.01),argc>5&&std::string(argv[5])=="closed",argc>6?argv[6]:"");
    result["preparation_s"]=prep;result["nq"]=plant.model.nq;result["nv"]=plant.model.nv;
    result["scenario"]=scenario;result["model"]=plant.meta;
    std::ofstream out(argv[4]);if(!out)throw std::runtime_error("cannot write result");out<<result.dump(2)<<'\n';
    std::cout<<det::Json({{"wall_s",result["wall_s"]},{"simulated_s",result["simulated_s"]},{"stop_reason",result["stop_reason"]},{"output",argv[4]}}).dump()<<'\n';
    return result["finite"].get<bool>()?0:3;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
