#include "solver/sts.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

SuperTimeStepper::SuperTimeStepper(const MeshInputs& in, int max_stages) : max_stages_(max_stages){
  F_ = AllocateFaceFluxes(in);
  const int ni = in.nx1_block + 2*in.nghost, nj = in.nx2_block + 2*in.nghost;
  y0_ = View4D("sts_y0", in.max_blocks, ni, nj, NVAR);
  ym2_ = View4D("sts_ym2", in.max_blocks, ni, nj, NVAR);
  l0_ = View4D("sts_l0", in.max_blocks, ni, nj, NVAR);
}

int SuperTimeStepper::StagesFor(Real ratio){
  const int s = (int)std::ceil((-1.0 + std::sqrt(9.0 + 16.0*ratio))/2.0 - 1e-12);
  return std::max(2, s);
}

Real SuperTimeStepper::MaxStep(Real dt_sts) const{
  const Real s = max_stages_;
  return 2.0*dt_sts*(s*s + s - 2.0)/4.0;
}

void SuperTimeStepper::Coefficients(int s, std::vector<Real>& mu, std::vector<Real>& nu, std::vector<Real>& mut,
                                    std::vector<Real>& gt, std::vector<Real>& c){
  const Real w1 = 4.0/(s*s + s - 2.0);
  std::vector<Real> b(s + 1, 1.0/3.0);
  for(int j = 3; j <= s; ++j) b[j] = (j*j + j - 2.0)/(2.0*j*(j + 1.0));
  mu.assign(s + 1, 0.0);
  nu.assign(s + 1, 0.0);
  mut.assign(s + 1, 0.0);
  gt.assign(s + 1, 0.0);
  mu[1] = 1.0;
  mut[1] = b[1]*w1;
  for(int j = 2; j <= s; ++j){
    mu[j] = (2.0*j - 1.0)/j*b[j]/b[j - 1];
    nu[j] = -(j - 1.0)/j*b[j]/b[j - 2];
    mut[j] = mu[j]*w1;
    gt[j] = -(1.0 - b[j - 1])*mut[j];
  }
  //flux weights: Y_j - Y_0 = tau sum_k d_j[k] L(Y_k), recursion of the stage formula
  std::vector<std::vector<Real>> d(s + 1, std::vector<Real>(s, 0.0));
  d[1][0] = mut[1];
  for(int j = 2; j <= s; ++j){
    for(int k = 0; k < s; ++k) d[j][k] = mu[j]*d[j - 1][k] + nu[j]*d[j - 2][k];
    d[j][j - 1] += mut[j];
    d[j][0] += gt[j];
  }
  c = d[s];
}

void SuperTimeStepper::Stage(Mesh& mesh, Real tau, Real mu, Real nu, Real mut, Real gt, bool first, Real drag){
  const int n_active = mesh.num_active_blocks();
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  View4D U = mesh.get_data_view();
  View1D bdx1 = mesh.get_block_dx1_view();
  View1D bdx2 = mesh.get_block_dx2_view();
  const View4D fx = F_.fx, fy = F_.fy, y0 = y0_, ym2 = ym2_, l0 = l0_;
  const Real c0 = 1.0 - mu - nu;
  Kokkos::parallel_for("RKL2Stage",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ng + nx1, ng + nx2}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const Real dx = bdx1(b), dy = bdx2(b);
      const int fi = i - ng, fj = j - ng;
      //the cell's current state, read before any component is overwritten
      Real q[NVAR];
      for(int k = 0; k < (int)NVAR; ++k) q[k] = U(b, i, j, k);
      for(int k = 0; k < (int)NVAR; ++k){
        //L of the current state: -div of the diffusive fluxes, plus the linear drag
        Real L = -(fx(b, fi + 1, fj, k) - fx(b, fi, fj, k))/dx - (fy(b, fi, fj + 1, k) - fy(b, fi, fj, k))/dy;
        if(k == IM1 || k == IM2) L -= drag*q[k];
        if(k == IEN) L -= drag*(q[IM1]*q[IM1] + q[IM2]*q[IM2])/q[IDN];
        if(first) l0(b, i, j, k) = L;
        const Real old = q[k];
        Real y = mu*old + mut*tau*L;
        if(nu != 0.0) y += nu*ym2(b, i, j, k);
        if(c0 != 0.0) y += c0*y0(b, i, j, k);
        if(gt != 0.0) y += gt*tau*l0(b, i, j, k);
        U(b, i, j, k) = y;
        ym2(b, i, j, k) = old; //Y_{j-1} becomes Y_{j-2} for the next stage
      }
    });
}

int SuperTimeStepper::Step(Mesh& mesh, Real tau, Real dt_sts, Real gamma, const DiffusionCoeffs& d, Real drag,
                           const WallFlags& walls, const std::function<void()>& fill_ghosts){
  const int s = StagesFor(tau/dt_sts);
  if(s > max_stages_) throw std::runtime_error("RKL2: " + std::to_string(s) + " stages needed, more than sts_max_stages");
  std::vector<Real> mu, nu, mut, gt, c;
  Coefficients(s, mu, nu, mut, gt, c);
  View4D reg = mesh.get_flux_view();
  if(acc_.extent(0) != reg.extent(0) || acc_.extent(2) != reg.extent(2))
    acc_ = View4D("sts_register", reg.extent(0), reg.extent(1), reg.extent(2), reg.extent(3));
  //the per-stage work touches only the active blocks (the views are sized for max_blocks)
  const auto act = Kokkos::make_pair(0, mesh.num_active_blocks());
  const auto all = Kokkos::ALL;
  const View4D acc = acc_;
  Kokkos::deep_copy(acc_, 0.0);
  Kokkos::deep_copy(Kokkos::subview(y0_, act, all, all, all), Kokkos::subview(mesh.get_data_view(), act, all, all, all));
  for(int j = 1; j <= s; ++j){
    //diffusive fluxes of Y_{j-1} (ghost cells are filled), into F_ and, at block edges, the register
    Kokkos::deep_copy(Kokkos::subview(F_.fx, act, all, all, all), 0.0);
    Kokkos::deep_copy(Kokkos::subview(F_.fy, act, all, all, all), 0.0);
    Kokkos::deep_copy(Kokkos::subview(reg, act, all, all, all), 0.0);
    AddDiffusiveFluxes(mesh, mesh.get_data_view(), gamma, d, F_, walls);
    const Real ck = c[j - 1];
    Kokkos::parallel_for("RKL2Register",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {act.second, (int)reg.extent(1),
                                                            (int)reg.extent(2), (int)reg.extent(3)}),
      KOKKOS_LAMBDA(const int b, const int f, const int t, const int k){ acc(b, f, t, k) += ck*reg(b, f, t, k); });
    Stage(mesh, tau, mu[j], nu[j], mut[j], gt[j], j == 1, drag);
    fill_ghosts();
  }
  //effective flux of the whole step -> refluxing at level boundaries, then ghost cells again
  Kokkos::deep_copy(reg, acc_);
  ApplyReflux(mesh, tau);
  fill_ghosts();
  return s;
}

Real StsTimeStep(Mesh& mesh, Real gamma, const DiffusionCoeffs& d, Real drag, Real cfl_diff){
  const Real tau_diff = DiffusiveTimeStep(mesh, gamma, d, 1.0);
  const Real inv = (d.active() ? 1.0/tau_diff : 0.0) + 0.5*drag;
  if(inv <= 0.0) return std::numeric_limits<Real>::max();
  return cfl_diff/inv;
}
