#pragma once

#include <functional>
#include <vector>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "solver/rhs.hpp"
#include "solver/diffusion.hpp"

/*
 RKL2 super-time-stepping, Meyer, Balsara & Aslam (2014, JCoPh 257, 594).

 With diffusion_integrator rkl2 the parabolic operator
   L(U) = -div F_diff(U) + drag(U),   drag = (0, -alpha m, -alpha |m|^2/rho) (optional, problem hook)
 leaves the hydro stages and is advanced by Strang splitting: RKL2(dt/2), hydro(dt), RKL2(dt/2).
 One RKL2 step of length tau uses s stages:
   Y_0 = U,  Y_1 = Y_0 + mut_1 tau L(Y_0),
   Y_j = mu_j Y_{j-1} + nu_j Y_{j-2} + (1 - mu_j - nu_j) Y_0 + mut_j tau L(Y_{j-1}) + gt_j tau L(Y_0),
 stable for tau lambda_max <= (s^2 + s - 2)/2 and second order. dt_sts = cfl_diff times the forward-Euler
 limit of L, so s is the smallest s >= 2 with (s^2 + s - 2)/4 >= tau/dt_sts; with cfl_diff = 0.5 every
 mode stays inside half the stable interval, where RKL2 damps it by 30-70% per step without sign changes.

 Conservation and refluxing: Y_s - Y_0 = tau sum_k c_k L(Y_k) with weights c_k (sum c_k = 1) fixed by the
 recursion, so the effective flux of the step is sum_k c_k F(Y_k). The flux register accumulates exactly
 that, and ApplyReflux(tau) corrects coarse cells at level boundaries once per RKL2 step, as the hydro
 integrators do with their stage-averaged register.
*/


class SuperTimeStepper{
public:
  SuperTimeStepper(const MeshInputs& in, int max_stages);

  //smallest s >= 2 with (s^2 + s - 2)/4 >= ratio
  static int StagesFor(Real ratio);
  //largest step whose two halves need at most max_stages stages each
  Real MaxStep(Real dt_sts) const;

  //advance the mesh data by tau; fill_ghosts after every stage (exchange + boundaries); returns s
  int Step(Mesh& mesh, Real tau, Real dt_sts, Real gamma, const DiffusionCoeffs& d, Real drag,
           const WallFlags& walls, const std::function<void()>& fill_ghosts);

  //per-stage coefficients (mu, nu, mut, gt) and the flux weights c_k for s stages
  static void Coefficients(int s, std::vector<Real>& mu, std::vector<Real>& nu, std::vector<Real>& mut,
                           std::vector<Real>& gt, std::vector<Real>& c);

  int max_stages() const { return max_stages_; }

  //one stage: data = mu data + nu ym2 + (1 - mu - nu) y0 + mut tau L(data) + gt tau l0, ym2 = old data;
  //with first = true L(data) is also stored in l0 (stage 1, data = Y_0)
  void Stage(Mesh& mesh, Real tau, Real mu, Real nu, Real mut, Real gt, bool first, Real drag);

private:
  int max_stages_;
  FaceFluxes F_;
  View4D y0_, ym2_, l0_, acc_;
};

//cfl_diff times the forward-Euler limit of diffusion plus drag: 1/(1/tau_diff + alpha/2)
Real StsTimeStep(Mesh& mesh, Real gamma, const DiffusionCoeffs& d, Real drag, Real cfl_diff);
