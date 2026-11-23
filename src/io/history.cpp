#include "io/history.hpp"
#include <iomanip>
#include <stdexcept>

//constructor
HistoryRecorder::HistoryRecorder(const std::string& outdir, int interval, Real gamma, Real nu, Real nu_bulk,
                                 const HistoryHook& hook)
  : interval_(interval), gamma_(gamma), nu_(nu), nub_(nu_bulk), hook_(hook){
  if(interval_ <= 0) return;
  file_.open(outdir + "/history.hst");
  if(!file_) throw std::runtime_error("Cannot open history file: " + outdir + "/history.hst");
  file_ << std::setprecision(17) << std::scientific;
}

//records the values for the hst file
void HistoryRecorder::record(Mesh& mesh, Real t, bool write_line){
  if(interval_ <= 0) return;
  const int n_active = mesh.num_active_blocks();
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  View4D U = mesh.get_data_view();
  View1D bdx1 = mesh.get_block_dx1_view();
  View1D bdx2 = mesh.get_block_dx2_view();
  const Real gm1 = gamma_ - 1.0, nu = nu_, nub = nub_;

  //state integrals and core rates, one pass over all interior cells
  Real mass = 0.0, mx = 0.0, my = 0.0, ekin = 0.0, etot = 0.0, enst = 0.0, dvisc = 0.0, wp = 0.0;
  Kokkos::parallel_reduce("HistoryIntegrals",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ng + nx1, ng + nx2}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& m, Real& px, Real& py, Real& ek,
                  Real& et, Real& en, Real& dv, Real& w){
      const Real dx = bdx1(b), dy = bdx2(b), dV = dx*dy;
      auto u = [&](int ii, int jj){ return U(b, ii, jj, IM1)/U(b, ii, jj, IDN); };
      auto v = [&](int ii, int jj){ return U(b, ii, jj, IM2)/U(b, ii, jj, IDN); };
      const Real rho = U(b, i, j, IDN), uc = u(i, j), vc = v(i, j);
      const Real ke = 0.5*rho*(uc*uc + vc*vc);
      const Real p = gm1*(U(b, i, j, IEN) - ke);
      //central velocity derivatives (ghost cells are filled at the top of a step)
      const Real ux = (u(i + 1, j) - u(i - 1, j))/(2.0*dx), uy = (u(i, j + 1) - u(i, j - 1))/(2.0*dy);
      const Real vx = (v(i + 1, j) - v(i - 1, j))/(2.0*dx), vy = (v(i, j + 1) - v(i, j - 1))/(2.0*dy);
      const Real div = ux + vy, om = vx - uy;
      m += rho*dV;
      px += U(b, i, j, IM1)*dV;
      py += U(b, i, j, IM2)*dV;
      ek += ke*dV;
      et += U(b, i, j, IEN)*dV;
      en += 0.5*om*om*dV;
      dv += rho*(nu*(2.0*(ux*ux + vy*vy) + (uy + vx)*(uy + vx) - (2.0/3.0)*div*div) + nub*div*div)*dV;
      w += p*div*dV;
    }, mass, mx, my, ekin, etot, enst, dvisc, wp);

  //problem rates (forcing, drag, cooling, ...)
  std::vector<HistoryTerm> extra = hook_ ? hook_(mesh, gamma_) : std::vector<HistoryTerm>();
  std::vector<Real> rates = {dvisc, wp};
  for(const auto& e : extra) rates.push_back(e.rate);

  if(first_){ //names, kinetic signs and the header, once
    rate_names_ = {"D_visc", "W_p"};
    kinetic_ = {-1, 1};
    for(const auto& e : extra){ rate_names_.push_back(e.name); kinetic_.push_back(e.kinetic); }
    integrals_.assign(rates.size(), 0.0);
    file_ << "# t dt mass mx my Ekin Eint Etot enstrophy";
    for(const auto& nm : rate_names_) file_ << " " << nm;
    for(const auto& nm : rate_names_) file_ << " int_" << nm;
    file_ << " eps_num\n";
    ekin0_ = ekin;
    t_prev_ = t;
  }else{ //trapezoidal rule from the previous sample, every step
         //TODO: might change later
    const Real h = t - t_prev_;
    for(size_t k = 0; k < rates.size(); ++k) integrals_[k] += 0.5*(rates_prev_[k] + rates[k])*h;
  }
  const Real dt = t - t_prev_;
  rates_prev_ = rates;
  t_prev_ = t;
  first_ = false;

  //numerical dissipation: what the kinetic-energy rates predict minus what actually happened
  Real predicted = 0.0;
  for(size_t k = 0; k < integrals_.size(); ++k) predicted += kinetic_[k]*integrals_[k];
  const Real eps = predicted - (ekin - ekin0_);

  if(!write_line) return;
  file_ << t << " " << dt << " " << mass << " " << mx << " " << my << " " << ekin << " " << etot - ekin
        << " " << etot << " " << enst;
  for(Real r : rates) file_ << " " << r;
  for(Real I : integrals_) file_ << " " << I;
  file_ << " " << eps << "\n";
  file_.flush();
}
