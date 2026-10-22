// Batch the existing frame/bounding-box calculation; no altered geometry.
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/serialization/model.hpp>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

using Vec = Eigen::VectorXd;
using V3 = Eigen::Vector3d;
using Mat = Eigen::MatrixXd;
using RowMat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
namespace {
// Closest-point regions match trimesh.triangles.closest_point, including its
// zero tolerance supplied by the Python caller.
V3 closest_triangle(const V3& p,const V3& a,const V3& b,const V3& c,double zero) {
  V3 ab=b-a,ac=c-a,ap=p-a;double d1=ab.dot(ap),d2=ac.dot(ap);
  if(d1<zero&&d2<zero)return a;
  V3 bp=p-b;double d3=ab.dot(bp),d4=ac.dot(bp);
  if(d3>-zero&&d4<=d3)return b;
  double vc=d1*d4-d3*d2;
  if(vc<zero&&d1>-zero&&d3<zero)return a+(d1/(d1-d3))*ab;
  V3 cp=p-c;double d5=ab.dot(cp),d6=ac.dot(cp);
  if(d6>-zero&&d5<=d6)return c;
  double vb=d5*d2-d1*d6;
  if(vb<zero&&d2>-zero&&d6<zero)return a+(d2/(d2-d6))*ac;
  double va=d3*d6-d5*d4;
  if(va<zero&&(d4-d3)>-zero&&(d5-d6)>-zero)return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);
  double denom=1./(va+vb+vc);return a+ab*(vb*denom)+ac*(vc*denom);
}
}
extern "C" void nearest_triangle_points(int count,const double* points,int nt,
    const double* triangles,double zero,double* output) {
  for(int i=0;i<count;++i) {
    V3 p=Eigen::Map<const V3>(points+3*i),best=V3::Zero();double distance=INFINITY;
    for(int j=0;j<nt;++j) {
      V3 candidate=closest_triangle(p,Eigen::Map<const V3>(triangles+9*j),
          Eigen::Map<const V3>(triangles+9*j+3),Eigen::Map<const V3>(triangles+9*j+6),zero);
      double d=(candidate-p).squaredNorm();
      if(d<distance) {distance=d;best=candidate;}
    }
    Eigen::Map<V3>(output+3*i)=best;
  }
}
extern "C" void paired_triangle_points(int count,const double* points,
    const double* triangles,double zero,double* output) {
  for(int i=0;i<count;++i)Eigen::Map<V3>(output+3*i)=closest_triangle(
      Eigen::Map<const V3>(points+3*i),Eigen::Map<const V3>(triangles+9*i),
      Eigen::Map<const V3>(triangles+9*i+3),Eigen::Map<const V3>(triangles+9*i+6),zero);
}
// Same bracketed multiplier solve as ObjectShape; pointwise work avoids
// repeatedly allocating tiny NumPy arrays in contact fitting.
extern "C" void ellipsoid_points(int count,const double* points,const double* axes,double* output) {
  const Eigen::Array3d a=Eigen::Map<const V3>(axes).array(),aa=a.square();
  Eigen::Index shortest;double amin=aa.minCoeff(&shortest);
  for(int i=0;i<count;++i) {
    Eigen::Array3d x=Eigen::Map<const V3>(points+3*i).array(),xa=x*a;
    auto value=[&](double l) { return (xa/(l+aa)).square().sum(); };
    double low=-amin+1e-15,high=std::max(xa.matrix().norm(),amin),l=0.;
    bool medial=value(low)<1.;
    for(int step=0;step<12;++step) {
      Eigen::Array3d denom=l+aa,terms=(xa/denom).square();
      double residual=terms.sum()-1.;
      if(residual>0.)low=l;else high=l;
      double derivative=-2.*(terms/denom).sum();
      double candidate=l-(std::abs(derivative)>1e-20?residual/derivative:0.);
      bool converged=std::abs(residual)<1e-12;
      if(!converged)l=(candidate>low&&candidate<high)?candidate:(low+high)/2.;
      if(converged||medial)break;
    }
    if(std::abs(value(l)-1.)>1e-10&&!medial) {
      for(int step=0;step<40;++step) {
        double middle=(low+high)/2.;if(value(middle)>1.)low=middle;else high=middle;
      }
      l=(low+high)/2.;
    }
    Eigen::Array3d p=aa*x/(l+aa);
    if(medial) {
      p.setZero();
      for(int j=0;j<3;++j)if(aa[j]>amin+1e-14)p[j]=aa[j]*x[j]/(aa[j]-amin);
      p[shortest]=a[shortest]*std::sqrt(std::max(0.,1.-(p/a).square().sum()));
    }
    Eigen::Map<V3>(output+3*i)=p.matrix();
  }
}
namespace {
thread_local std::string last_error;
struct Part { int frame; V3 center,half; };
struct Box { V3 center,half; };
struct Context {
  pinocchio::Model model;
  std::unique_ptr<pinocchio::Data> data;
  std::vector<int> frames;
  std::vector<Part> parts;
  std::vector<Box> obstacles;
  bool floor;
  double floor_height;
};
double sign(double x) { return (x>0.)-(x<0.); }
}

extern "C" const char* geometry_error() { return last_error.c_str(); }
extern "C" void geometry_destroy(void* pointer) { delete static_cast<Context*>(pointer); }
extern "C" void* geometry_create(const char* serialized,int nf,const int* frames,
    int nb,const int* part_frames,const double* centers,const double* halves,
    int no,const double* obstacles,int floor,double floor_height) {
  try {
    auto c=std::make_unique<Context>();
    c->model.loadFromString(std::string(serialized));
    c->data=std::make_unique<pinocchio::Data>(c->model);
    c->frames.assign(frames,frames+nf);
    for(int i=0;i<nb;++i)c->parts.push_back({part_frames[i],Eigen::Map<const V3>(centers+3*i),Eigen::Map<const V3>(halves+3*i)});
    for(int i=0;i<no;++i)c->obstacles.push_back({Eigen::Map<const V3>(obstacles+6*i),Eigen::Map<const V3>(obstacles+6*i+3)});
    c->floor=floor;c->floor_height=floor_height;
    for(int frame:c->frames)if(frame<0||frame>=int(c->model.nframes))throw std::runtime_error("invalid contact frame");
    for(const auto& p:c->parts)if(p.frame<0||p.frame>=int(c->model.nframes))throw std::runtime_error("invalid part frame");
    return c.release();
  } catch(const std::exception& e) { last_error=e.what();return nullptr; }
}

extern "C" int geometry_evaluate(void* pointer,const double* configuration,
    const double* tangent_map,int nx,double* placements,double* jacobians,
    double* residuals,double* derivatives) {
  try {
    auto& c=*static_cast<Context*>(pointer);auto& d=*c.data;
    Eigen::Map<const Vec> q(configuration,c.model.nq);
    Eigen::Map<const RowMat> lift(tangent_map,c.model.nv,nx);
    if(!q.allFinite()||!lift.allFinite())throw std::runtime_error("nonfinite geometry input");
    pinocchio::computeJointJacobians(c.model,d,q);
    pinocchio::updateFramePlacements(c.model,d);
    Mat full=Mat::Zero(6,c.model.nv),J(6,nx);
    auto frame_jacobian=[&](int frame) {
      full.setZero();
      pinocchio::getFrameJacobian(c.model,d,frame,pinocchio::LOCAL_WORLD_ALIGNED,full);
      J.noalias()=full*lift;
    };
    for(size_t i=0;i<c.frames.size();++i) {
      const auto& T=d.oMf[c.frames[i]];
      for(int row=0;row<3;++row) {
        placements[12*i+row]=T.translation()[row];
        for(int col=0;col<3;++col)placements[12*i+3+3*row+col]=T.rotation()(row,col);
      }
      frame_jacobian(c.frames[i]);
      Eigen::Map<RowMat>(jacobians+6*nx*i,6,nx)=J;
    }
    int columns=int(c.obstacles.size())+int(c.floor);
    Eigen::Map<Vec> r(residuals,c.parts.size()*columns);r.setZero();
    Eigen::Map<RowMat> D(derivatives,r.size(),nx);D.setZero();
    std::vector<double> overlaps(columns),signs(columns);
    std::vector<int> axes(columns);
    Mat center_jac(3,nx),half_jac(3,nx);
    for(size_t i=0;i<c.parts.size();++i) {
      const auto& part=c.parts[i];const auto& T=d.oMf[part.frame];
      V3 center=T.act(part.center),half=T.rotation().cwiseAbs()*part.half;
      int offset=0;bool active=false;
      if(c.floor) { overlaps[0]=c.floor_height-center[2]+half[2];axes[0]=2;signs[0]=1.;offset=1; }
      for(size_t j=0;j<c.obstacles.size();++j) {
        V3 delta=center-c.obstacles[j].center;
        V3 depth=half+c.obstacles[j].half+V3::Constant(.035)-delta.cwiseAbs();
        Eigen::Index axis;overlaps[offset+j]=depth.minCoeff(&axis);
        axes[offset+j]=int(axis);signs[offset+j]=sign(delta[axis]);
      }
      for(int j=0;j<columns;++j) {
        if(overlaps[j]>0.) { r[i*columns+j]=2*overlaps[j];active=true; }
      }
      if(!active)continue;
      frame_jacobian(part.frame);
      center_jac=J.topRows(3)-pinocchio::skew(T.rotation()*part.center)*J.bottomRows(3);
      Eigen::Matrix3d angular=Eigen::Matrix3d::Zero();
      for(int col=0;col<3;++col) {
        Eigen::Matrix3d cross=pinocchio::skew(T.rotation().col(col));
        for(int row=0;row<3;++row)angular.row(row)-=part.half[col]*sign(T.rotation()(row,col))*cross.row(row);
      }
      half_jac=angular*J.bottomRows(3);
      for(int j=0;j<columns;++j)if(overlaps[j]>0.)
        D.row(i*columns+j)=2*(half_jac.row(axes[j])-signs[j]*center_jac.row(axes[j]));
    }
    return 0;
  } catch(const std::exception& e) { last_error=e.what();return -1; }
}

namespace {
// Same signed nearest-surface query and finite difference as ObjectShape.
void primitive_surface(int shape,const V3& x,const V3& a,double& gap,V3& normal,
    int ne,const double* equations,int nt,const double* triangles,double triangle_zero) {
  V3 p;bool inside;
  if(shape==1) {
    p=x.cwiseMax(-a).cwiseMin(a);inside=(x.array().abs()<=a.array()).all();
    if(inside) {Eigen::Index i;(a-x.cwiseAbs()).minCoeff(&i);p[i]=(x[i]>=0.?1.:-1.)*a[i];}
  } else if(shape==2) {
    double r=x.head<2>().norm();Eigen::Vector2d direction(1.,0.);
    if(r>1e-14)direction=x.head<2>()/r;
    inside=r<=a[0]&&std::abs(x[2])<=a[2];p=x;
    p.head<2>()=direction*std::min(r,a[0]);p[2]=std::clamp(x[2],-a[2],a[2]);
    if(inside) {
      if(a[0]-r<a[2]-std::abs(x[2]))p.head<2>()=direction*a[0];
      else p[2]=(x[2]>=0.?1.:-1.)*a[2];
    }
  } else if(shape==3) {
    inside=(x.array()/a.array()).square().sum()<1.;
    ellipsoid_points(1,x.data(),a.data(),p.data());
  } else {
    double largest=-INFINITY;int face=0;
    for(int i=0;i<ne;++i) {
      double value=Eigen::Map<const V3>(equations+4*i).dot(x)+equations[4*i+3];
      if(value>largest) {largest=value;face=i;}
    }
    inside=largest<0.;p=x-largest*Eigen::Map<const V3>(equations+4*face);
    bool outside=false;
    for(int i=0;i<ne;++i)outside|=Eigen::Map<const V3>(equations+4*i).dot(p)+equations[4*i+3]>1e-12;
    if(outside) {
      double nearest=INFINITY;
      for(int i=0;i<nt;++i) {
        V3 candidate=closest_triangle(x,Eigen::Map<const V3>(triangles+9*i),
            Eigen::Map<const V3>(triangles+9*i+3),Eigen::Map<const V3>(triangles+9*i+6),triangle_zero);
        double distance=(candidate-x).squaredNorm();
        if(distance<nearest) {nearest=distance;p=candidate;}
      }
    }
  }
  double distance=(x-p).norm();gap=inside?-distance:distance;
  if(distance>1e-12) {normal=(x-p)/gap;return;}
  V3 u=p.normalized();normal.setZero();
  if(shape==1) {
    V3 distances;
    for(int i=0;i<3;++i)distances[i]=std::abs(u[i])>1e-12?a[i]/std::abs(u[i]):INFINITY;
    Eigen::Index i;distances.minCoeff(&i);normal[i]=sign(u[i]);
  } else if(shape==2) {
    double side=a[0]/std::max(u.head<2>().norm(),1e-15),cap=a[2]/std::max(std::abs(u[2]),1e-15);
    if(side<cap)normal.head<2>()=side*u.head<2>()/a[0];else normal[2]=sign(u[2]);
  } else if(shape==3)normal=(p.array()/a.array().square()).matrix().normalized();
  else {
    double nearest=INFINITY;
    for(int i=0;i<ne;++i) {
      V3 n=Eigen::Map<const V3>(equations+4*i);double denom=n.dot(u);
      double distance=denom>1e-10?-equations[4*i+3]/denom:INFINITY;
      if(distance<nearest) {nearest=distance;normal=n;}
    }
  }
}
}

extern "C" int geometry_fit_evaluate(void* pointer,const double* configuration,
    const double* tangent_map,int nx,int count,const double* target_points,
    const double* target_normals,int shape,const double* half_size,double contact_gap,
    int ne,const double* equations,int nt,const double* triangles,double triangle_zero,
    const double* delta,double* residuals,double* derivatives) {
  try {
    auto& c=*static_cast<Context*>(pointer);
    if(shape<0||shape>4||int(c.frames.size())!=count+int(shape!=0)||(shape==4&&(!ne||!nt)))
      throw std::runtime_error("invalid fused fit shape/frames");
    int nb=c.parts.size()*(c.obstacles.size()+int(c.floor));
    std::vector<double> poses(12*c.frames.size()),jac(6*nx*c.frames.size()),br(nb),bj(nb*nx);
    int status=geometry_evaluate(pointer,configuration,tangent_map,nx,poses.data(),jac.data(),br.data(),bj.data());
    if(status)return status;
    int rows=count*(shape?7:6),total=rows+nb+nx;
    Eigen::Map<Vec> r(residuals,total);Eigen::Map<RowMat> D(derivatives,total,nx);D.setZero();
    Eigen::Matrix3d R=Eigen::Matrix3d::Identity();V3 object=V3::Zero(),axes=V3::Ones();
    if(shape) {R=Eigen::Map<const RowMat>(poses.data()+12*count+3,3,3);object=Eigen::Map<const V3>(poses.data()+12*count);axes=Eigen::Map<const V3>(half_size);}
    for(int i=0;i<count;++i) {
      V3 position=Eigen::Map<const V3>(poses.data()+12*i);
      Eigen::Matrix3d rotation=Eigen::Map<const RowMat>(poses.data()+12*i+3,3,3);
      V3 normal=rotation.col(0),target=Eigen::Map<const V3>(target_points+3*i);
      Eigen::Map<const RowMat> J(jac.data()+6*nx*i,6,nx);
      Mat normal_jac=-pinocchio::skew(normal)*J.bottomRows(3);
      if(!shape) {
        r.segment<3>(6*i)=position-target;
        r.segment<3>(6*i+3)=.15*(normal-Eigen::Map<const V3>(target_normals+3*i));
        D.middleRows(6*i,3)=J.topRows(3);D.middleRows(6*i+3,3)=.15*normal_jac;
      } else {
        V3 local=R.transpose()*(position-object),outward,gradient;Eigen::Matrix3d dn;
        double gap;primitive_surface(shape,local,axes,gap,outward,ne,equations,nt,triangles,triangle_zero);
        for(int axis=0;axis<3;++axis) {
          V3 plus=local,minus=local,np,nm;double gp,gm;plus[axis]+=1e-6;minus[axis]-=1e-6;
          primitive_surface(shape,plus,axes,gp,np,ne,equations,nt,triangles,triangle_zero);
          primitive_surface(shape,minus,axes,gm,nm,ne,equations,nt,triangles,triangle_zero);
          gradient[axis]=(gp-gm)/2e-6;dn.col(axis)=(np-nm)/2e-6;
        }
        Mat local_jac=R.transpose()*J.topRows(3);
        r[7*i]=gap-contact_gap;r.segment<3>(7*i+1)=.15*(normal+R*outward);
        r.segment<3>(7*i+4)=.03*(position-target);
        D.row(7*i)=gradient.transpose()*local_jac;
        D.middleRows(7*i+1,3)=.15*(normal_jac+R*dn*local_jac);
        D.middleRows(7*i+4,3)=.03*J.topRows(3);
      }
    }
    r.segment(rows,nb)=Eigen::Map<Vec>(br.data(),nb);
    D.middleRows(rows,nb)=Eigen::Map<RowMat>(bj.data(),nb,nx);
    r.tail(nx)=.0001*Eigen::Map<const Vec>(delta,nx);
    D.bottomRows(nx).diagonal().setConstant(.0001);
    return 0;
  } catch(const std::exception& e) {last_error=e.what();return -1;}
}
