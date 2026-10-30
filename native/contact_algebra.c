// Independent contact projections and branch assembly. No relaxed constraints.
#include <math.h>
#include <float.h>
#include <string.h>
void contact_project(int count,const double* affine,const double* rn,double ratio,
    const double* operators,const double* dual,const double* G,int* modes,double* output){
  for(int i=0;i<count;i++){
    const double* a=affine+3*i;double best=rn[i]*(a[0]*a[0]+(a[1]*a[1]+a[2]*a[2])/ratio);
    modes[i]=-1;double* out=output+3*i;out[0]=out[1]=out[2]=0.;
    for(int mode=0;mode<17;mode++){
      double f[3]={0,0,0},mu[2]={0,0};
      for(int j=0;j<3;j++)for(int k=0;k<3;k++)f[j]+=operators[mode*9+j*3+k]*a[k];
      for(int j=0;j<2;j++)for(int k=0;k<3;k++)mu[j]+=dual[mode*6+j*3+k]*a[k];
      if(rn[i]*mu[0]<-1e-9||rn[i]*mu[1]<-1e-9||f[0]<-1e-8)continue;
      int valid=1;for(int j=0;j<8;j++){double v=0;for(int k=0;k<3;k++)v+=G[j*3+k]*f[k];if(v>1e-8){valid=0;break;}}
      if(!valid)continue;
      double cost=0;for(int j=0;j<3;j++){double v=f[j]-a[j];cost+=v*v*(j?rn[i]/ratio:rn[i]);}
      if(cost<best){best=cost;modes[i]=mode;memcpy(out,f,3*sizeof(double));}
    }
  }
}

void contact_branch(int rows,int n,int nv,int nu,int count,int ne,int ni,
    const double* A,const double* b,const double* pressure,const double* Dn,
    const double* p0,const double* rn,const double* alpha,double ratio,
    const double* H,const double* vertices,double C,const int* modes,
    double* out,double* bound){
  int multipliers=0,apex=0;for(int i=0;i<count;i++){int m=modes[i];if(m<0)apex++;else if(m>0)multipliers+=m>8?2:1;}
  int total=n+count+multipliers,eq=ne+count+multipliers,ineq=ni+8*apex+multipliers;
  int fullrows=rows+count+2*multipliers+8*apex;
  memset(out,0,(size_t)fullrows*total*sizeof(double));
  int retained=0;
  for(int r=0;r<rows;r++){
    int contact=(r>=nv&&r<nv+2*count)?(r-nv)/2:-1;
    if(r<ne&&contact>=0&&modes[contact]<0)continue;
    int target=r<ne?retained++:r<ne+ni?eq+r-ne:eq+ineq+r-ne-ni;
    double factor=contact>=0?rn[contact]/ratio/alpha[2*contact]:1.;
    for(int j=0;j<n;j++)out[target*total+j]=A[r*n+j]*factor;
    for(int j=0;j<count;j++)out[target*total+n+j]=pressure[r*count+j]*factor;
    bound[target]=b[r]*factor;
  }
  int e=retained,u=eq+ni,offset=n+count;
  for(int i=0;i<count;i++){
    int mode=modes[i],r0=nv+2*i,col=nv+nu+2*i;
    double factor=rn[i]/ratio/alpha[2*i];
    if(mode<0){
      for(int j=0;j<nv;j++)out[e*total+j]=-Dn[i*nv+j];
      out[e*total+n+i]=1;bound[e++]=-p0[i];
      out[e*total+col]=1;bound[e++]=0;out[e*total+col+1]=1;bound[e++]=0;
      for(int v=0;v<8;v++){
        for(int j=0;j<nv;j++)out[u*total+j]=-rn[i]*Dn[i*nv+j]-factor*(vertices[2*v]*A[r0*n+j]+vertices[2*v+1]*A[(r0+1)*n+j]);
        bound[u++]=-rn[i]*p0[i]-factor*(vertices[2*v]*b[r0]+vertices[2*v+1]*b[r0+1]);
      }
    }else{
      int faces=mode==0?0:mode<=8?1:2;int face0=mode<=8?mode-1:mode-9;
      int target=r0;for(int j=0;j<i;j++)if(modes[j]<0)target-=2;
      int normal=e+faces;out[normal*total+n+i]=rn[i];bound[normal]=0;
      for(int f=0;f<faces;f++){
        int face=(face0+f)%8;
        out[target*total+offset]=H[2*face];out[(target+1)*total+offset]=H[2*face+1];out[normal*total+offset]=-C;
        for(int j=0;j<nv;j++)out[e*total+j]=C*Dn[i*nv+j];
        out[e*total+col]=H[2*face];out[e*total+col+1]=H[2*face+1];out[e*total+n+i]=-C;bound[e++]=C*p0[i];
        out[u*total+offset++]=-1;bound[u++]=0;
      }
      e++;
    }
  }
}
