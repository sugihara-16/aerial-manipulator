// Same bounded-variable least-squares steps as SciPy 1.15.3 optimize/_lsq/bvls.py.
// SciPy's BSD license is retained in SCIPY_LICENSE.txt. Uses Eigen QR for full-rank
// dense subproblems; unsupported/rank-deficient problems return to Python.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
using Vec=Eigen::VectorXd;
using Mat=Eigen::MatrixXd;
using RowMat=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
static Vec least_squares(const Mat& A,const Vec& b) {
  if(A.cols()==0)return Vec(0);
  Eigen::ColPivHouseholderQR<Mat> qr(A);
  qr.setThreshold(std::numeric_limits<double>::epsilon()*std::max(A.rows(),A.cols()));
  if(qr.rank()!=A.cols())throw std::runtime_error("rank");
  return qr.solve(b);
}
static double optimality(const Vec& gradient,const Vec& bound) {
  double result=-std::numeric_limits<double>::infinity();
  for(int i=0;i<gradient.size();i++)result=std::max(result,bound[i]==0?std::abs(gradient[i]):gradient[i]*bound[i]);
  return result;
}
extern "C" int bounded_solve(int m,int n,const double* aa,const double* bb,
    const double* lower,const double* upper,double tol,int maxiter,
    double* solution,double* residual,double* bound_output,double* unbounded,double* stats) {
  try {
    Mat A=Eigen::Map<const RowMat>(aa,m,n);
    Vec b=Eigen::Map<const Vec>(bb,m),lb=Eigen::Map<const Vec>(lower,n),ub=Eigen::Map<const Vec>(upper,n);
    Vec x=least_squares(A,b),bound=Vec::Zero(n);Eigen::Map<Vec>(unbounded,n)=x;
    bool feasible=true;
    for(int i=0;i<n;i++)feasible=feasible&&x[i]>=lb[i]&&x[i]<=ub[i];
    int status=feasible?3:0,iteration=0;
    Vec r=A*x-b,g=A.transpose()*r;
    double cost=.5*r.squaredNorm(),opt=g.cwiseAbs().maxCoeff();
    if(!feasible) {
      for(int i=0;i<n;i++) {
        if(x[i]<=lb[i]){x[i]=lb[i];bound[i]=-1.;}
        if(x[i]>=ub[i]){x[i]=ub[i];bound[i]=1.;}
      }
      auto free_solve=[&](std::vector<int>& free) {
        free.clear();Vec fixed=x;
        for(int i=0;i<n;i++)if(bound[i]==0){free.push_back(i);fixed[i]=0;}
        Mat sub(m,free.size());for(int j=0;j<(int)free.size();j++)sub.col(j)=A.col(free[j]);
        return Vec(least_squares(sub,b-A*fixed));
      };
      r=A*x-b;cost=.5*r.squaredNorm();g=A.transpose()*r;
      for(;;) {
        bool any=false;for(int i=0;i<n;i++)any=any||bound[i]==0;
        if(!any)break;
        iteration++;std::vector<int> free;Vec z=free_solve(free);bool violated=false;
        for(int j=0;j<(int)free.size();j++) {
          int i=free[j];x[i]=z[j];
          if(z[j]<lb[i]){x[i]=lb[i];bound[i]=-1.;violated=true;}
          if(z[j]>ub[i]){x[i]=ub[i];bound[i]=1.;violated=true;}
        }
        r=A*x-b;cost=.5*r.squaredNorm();g=A.transpose()*r;
        if(!violated)break;
      }
      maxiter+=iteration;opt=optimality(g,bound);int termination=0;
      for(;iteration<maxiter;iteration++) {
        if(opt<tol)termination=1;
        if(termination){status=termination;break;}
        Eigen::Index index;(g.array()*bound.array()).maxCoeff(&index);bound[index]=0;
        for(int inner=0;;inner++) {
          if(inner>4*n+20)throw std::runtime_error("inner limit");
          std::vector<int> free;Vec z=free_solve(free);
          double alpha=std::numeric_limits<double>::infinity();int hit=-1;double side=0;
          // Lower-bound candidates precede upper-bound candidates, as in SciPy.
          for(int sign:{-1,1})for(int j=0;j<(int)free.size();j++) {
            int i=free[j];bool violates=sign<0?z[j]<lb[i]:z[j]>ub[i];
            if(violates) {
              double value=((sign<0?lb[i]:ub[i])-x[i])/(z[j]-x[i]);
              if(value<alpha){alpha=value;hit=i;side=sign;}
            }
          }
          if(hit<0){for(int j=0;j<(int)free.size();j++)x[free[j]]=z[j];break;}
          for(int j=0;j<(int)free.size();j++)x[free[j]]=(1-alpha)*x[free[j]]+alpha*z[j];
          bound[hit]=side;
        }
        r=A*x-b;double next=.5*r.squaredNorm();
        if(cost-next<tol*cost)termination=2;
        cost=next;g=A.transpose()*r;opt=optimality(g,bound);
      }
      if(!status)status=termination;
      if(iteration==maxiter)iteration--; // Match Python's last loop index.
      iteration++;
    }
    Eigen::Map<Vec>(solution,n)=x;Eigen::Map<Vec>(residual,m)=r;Eigen::Map<Vec>(bound_output,n)=bound;
    stats[0]=cost;stats[1]=opt;stats[2]=iteration;
    return status;
  }catch(...){return -1;}
}
