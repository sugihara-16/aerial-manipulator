// Rotation-minimizing grasp frame for an exactly axisymmetric shape.
#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>
extern "C" void transport_frame(int axis, const double *old,
                                const double *observed, const double *omega,
                                const double *alpha, double *frame,
                                double *velocity, double *acceleration) {
  double a[3], b[3], v[3], S[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int j = 0; j < 3; ++j) {
    a[j] = old[3 * j + axis];
    b[j] = observed[3 * j + axis];
  }
  double c = mju_dot3(a, b);
  mju_cross(v, a, b);
  if (c > -1. + 1e-10) {
    double K[9] = {0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}, K2[9];
    mju_mulMatMat(K2, K, K, 3, 3, 3);
    for (int j = 0; j < 9; ++j)
      S[j] += K[j] + K2[j] / (1. + c);
  } else { // The antipodal representative is non-unique; choose an old
           // transverse axis.
    double t[3];
    for (int j = 0; j < 3; ++j)
      t[j] = old[3 * j + (axis + 1) % 3];
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        S[3 * j + k] = 2 * t[j] * t[k] - (j == k ? 1. : 0.);
  }
  mju_mulMatMat(frame, S, old, 3, 3, 3);
  double spin = mju_dot3(b, omega), angular = mju_dot3(b, alpha), bdot[3];
  mju_cross(bdot, omega, b);
  for (int j = 0; j < 3; ++j) {
    velocity[j] = omega[j] - b[j] * spin;
    acceleration[j] = alpha[j] - b[j] * angular - bdot[j] * spin;
  }
}

#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>
#include <vector>
extern "C" void opening_ik(const mjModel *m, mjData *d, int np,
                           const int *bodies, const double *local,
                           const double *target, const double *W, int nc,
                           const int *cols, const double *scales, int nj,
                           const int *qi, const double *ranges, double *delta) {
  std::vector<double> initial(d->qpos, d->qpos + m->nq), J(3 * np * nc),
      rhs(3 * np), jac(3 * m->nv), H(nc * nc), b(nc), x(nc), dq(m->nv);
  for (int it = 0; it < 4; ++it) {
    mj_kinematics(m, d);
    mj_comPos(m, d);
    for (int k = 0; k < np; ++k) {
      double pos[3], err[3];
      mju_mulMatVec(pos, d->xmat + 9 * bodies[k], local + 3 * k, 3, 3);
      mju_addTo3(pos, d->xpos + 3 * bodies[k]);
      mju_sub3(err, target + 3 * k, pos);
      mju_mulMatVec(rhs.data() + 3 * k, W + 9 * k, err, 3, 3);
      mj_jac(m, d, jac.data(), nullptr, pos, bodies[k]);
      for (int a = 0; a < 3; ++a)
        for (int j = 0; j < nc; ++j) {
          double v = 0.;
          for (int z = 0; z < 3; ++z)
            v += W[9 * k + 3 * a + z] * jac[z * m->nv + cols[j]];
          J[(3 * k + a) * nc + j] = v * scales[j];
        }
    }
    std::fill(H.begin(), H.end(), 0.);
    std::fill(b.begin(), b.end(), 0.);
    for (int i = 0; i < nc; ++i) {
      H[i * nc + i] = .0001;
      for (int r = 0; r < 3 * np; ++r) {
        double a = J[r * nc + i];
        b[i] += a * rhs[r];
        for (int j = 0; j <= i; ++j)
          H[i * nc + j] += a * J[r * nc + j];
      }
    }
    mju_cholFactor(H.data(), nc, 1e-12);
    mju_cholSolve(x.data(), H.data(), b.data(), nc);
    std::fill(dq.begin(), dq.end(), 0.);
    for (int j = 0; j < nc; ++j) {
      double limit = j < 3 ? .02 : .15;
      dq[cols[j]] = std::clamp(x[j] * scales[j], -limit, limit);
    }
    mj_integratePos(m, d->qpos, dq.data(), 1.);
    for (int j = 0; j < nj; ++j)
      d->qpos[qi[j]] =
          std::clamp(d->qpos[qi[j]], ranges[2 * j], ranges[2 * j + 1]);
  }
  mj_differentiatePos(m, delta, 1., initial.data(), d->qpos);
  mju_copy(d->qpos, initial.data(), m->nq);
  mj_kinematics(m, d);
  mj_comPos(m, d);
}
