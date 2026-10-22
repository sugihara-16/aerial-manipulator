// Diagnostic native stepping/output collection. No new physics or controller.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>

extern "C" int rollout_version() { return mjVERSION_HEADER; }

extern "C" int contact_rollout(const mjModel* m, mjData* d, int steps,
                               int object, int pads, const int* group,
                               const int* bodies, double* states,
                               double* loads, double* motors, double* penetration,
                               double* poses) {
  if (!m || !d || steps < 1 || object < 0 || object >= m->ngeom || pads < 1)
    return -1;
  const int stride = 1 + m->nq + m->nv;
  for (int k = 0; k < steps; ++k) {
    double before = d->time;
    mj_step(m, d);
    if (std::abs(d->time - before - m->opt.timestep) > 1e-10) return -2;
    double* state = states + k * stride;
    state[0] = d->time;
    std::copy_n(d->qpos, m->nq, state + 1);
    std::copy_n(d->qvel, m->nv, state + 1 + m->nq);
    for (int i = 0; i < stride; ++i) if (!std::isfinite(state[i])) return -3;
    double* output = loads + k * pads * 7;
    std::fill_n(output, pads * 7, 0.);
    penetration[k] = 0.;
    for (int c = 0; c < d->ncon; ++c) {
      const auto& contact = d->contact[c];
      penetration[k] = std::max(penetration[k], -contact.dist);
      bool first = contact.geom[0] == object;
      if ((!first && contact.geom[1] != object) || contact.efc_address < 0)
        continue;
      int pad = group[contact.geom[first ? 1 : 0]];
      if (pad < 0) continue;
      if (pad >= pads || contact.dim != 3) return -4;
      double local[6], force[3], offset[3], moment[3];
      mj_contactForce(m, d, c, local);
      mju_mulMatTVec(force, contact.frame, local, 3, 3);
      mju_scl3(force, force, first ? 1. : -1.);
      mju_sub3(offset, contact.pos, d->xpos + 3 * bodies[pad]);
      mju_cross(moment, offset, force);
      output[pad * 7] += local[0];
      for (int i = 0; i < 3; ++i) {
        output[pad * 7 + 1 + i] += force[i];
        output[pad * 7 + 4 + i] += moment[i];
      }
    }
    for (int i = 0; i < m->nu; ++i) {
      motors[(k * 2) * m->nu + i] = d->actuator_force[i];
      motors[(k * 2 + 1) * m->nu + i] =
          m->actuator_gainprm[i * mjNGAIN] * d->ctrl[i] +
          m->actuator_biasprm[i * mjNBIAS + 1] * d->actuator_length[i] +
          m->actuator_biasprm[i * mjNBIAS + 2] * d->actuator_velocity[i];
    }
    if (poses) {
      // mj_step integrates qpos after evaluating forces. Recompute kinematics
      // so these transforms correspond to the saved ENDPOINT qpos, not q(k).
      mj_kinematics(m, d);
      for (int j = 0; j <= pads; ++j) {
        int body = j == pads ? m->geom_bodyid[object] : bodies[j];
        double* pose = poses + (k * (pads + 1) + j) * 7;
        std::copy_n(d->xpos + 3 * body, 3, pose);
        std::copy_n(d->xquat + 4 * body, 4, pose + 3);
      }
    }
  }
  return 0;
}
