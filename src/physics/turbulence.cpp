#include "physics/turbulence.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

std::vector<std::array<int, 2>> TurbulenceModeList(int k_low, Real k_peak, int k_high){
  std::vector<std::array<int, 2>> out;
  for(int i = 0; i <= k_high; ++i){
    for(int j = -k_high; j <= k_high; ++j){
      const Real k = std::sqrt(Real(i*i + j*j));
      if(k > k_high || k < k_low) continue;       //outside the shell
      const Real q = (k/k_peak)*(k/k_peak);
      if(q*(2.0 - q) < 0.0) continue;             //spectral shape must be >= 0
      if(i == 0 && j < 0) continue;               //(0, -j) is the partner of (0, j)
      out.push_back({i, j});
    }
  }
  return out;
}

Real TurbulenceDriver::uniform01(){
  //53 random bits -> [0, 1): the raw mt19937_64 sequence is fixed by the C++ standard
  return (rng_() >> 11) * (1.0/9007199254740992.0);
}

Real TurbulenceDriver::gaussian(){
  if(have_spare_){ have_spare_ = false; return spare_; }
  Real u1 = uniform01(), u2 = uniform01();
  if(u1 < 1e-300) u1 = 1e-300;
  const Real r = std::sqrt(-2.0*std::log(u1)), th = 2.0*M_PI*u2;
  spare_ = r*std::sin(th); have_spare_ = true;
  return r*std::cos(th);
}

TurbulenceDriver::TurbulenceDriver(const TurbParams& p, Real x0, Real y0, Real Lx, Real Ly)
  : p_(p), x0_(x0), y0_(y0), rng_(p.seed){
  auto require = [](bool ok, const std::string& msg){
    if(!ok) throw std::runtime_error("Parameter error: " + msg);
  };
  require(p.k_low >= 1, "turb_k_low must be >= 1 (k = 0 would be a uniform push)");
  require(p.k_high >= p.k_low, "turb_k_high must be >= turb_k_low");
  require(p.k_peak > 0.0, "turb_k_peak must be positive");
  require(p.tcorr > 0.0, "turb_tcorr must be positive");
  require(p.accel_rms >= 0.0, "turb_accel_rms must be >= 0");
  require(p.sol_weight >= 0.0 && p.sol_weight <= 1.0, "turb_sol_weight must be in [0, 1]");

  std::vector<std::array<int, 2>> all = TurbulenceModeList(p.k_low, p.k_peak, p.k_high);
  require(!all.empty(), "no turbulence modes between turb_k_low and turb_k_high");
  if(p.nmodes == 0 || p.nmodes == (int)all.size()){
    modes_ = all;
  }else{
    require(p.nmodes > 0 && p.nmodes <= (int)all.size(),
            "turb_nmodes is " + std::to_string(p.nmodes) + " but the shell has only " +
            std::to_string(all.size()) + " modes");
    //random subset (Fisher-Yates with our own uniform numbers), kept in list order
    std::vector<int> idx(all.size());
    for(size_t n = 0; n < idx.size(); ++n) idx[n] = (int)n;
    for(int n = (int)idx.size() - 1; n > 0; --n){
      int m = (int)(uniform01()*(n + 1));
      if(m > n) m = n;
      std::swap(idx[n], idx[m]);
    }
    idx.resize(p.nmodes);
    std::sort(idx.begin(), idx.end());
    for(int n : idx) modes_.push_back(all[n]);
  }

  const int nm = (int)modes_.size();
  d_modes_ = TurbModesView("turb_modes", nm);
  h_modes_ = Kokkos::create_mirror_view(d_modes_);
  const Real zeta = p.sol_weight;
  for(int n = 0; n < nm; ++n){
    const Real kx = 2.0*M_PI*modes_[n][0]/Lx, ky = 2.0*M_PI*modes_[n][1]/Ly;
    const Real kn = std::sqrt(Real(modes_[n][0]*modes_[n][0] + modes_[n][1]*modes_[n][1]));
    const Real q = (kn/p.k_peak)*(kn/p.k_peak);
    h_modes_(n, 0) = kx;
    h_modes_(n, 1) = ky;
    h_modes_(n, 2) = q*(2.0 - q); //spectral shape as amplitude weight
    //start in the stationary distribution: projected unit-variance noise
    const Real k2 = kx*kx + ky*ky;
    for(int part = 0; part < 2; ++part){ //real and imaginary part
      const Real ex = gaussian(), ey = gaussian();
      const Real kdote = (kx*ex + ky*ey)/k2;
      h_modes_(n, 3 + part) = zeta*ex + (1.0 - 2.0*zeta)*kx*kdote;
      h_modes_(n, 5 + part) = zeta*ey + (1.0 - 2.0*zeta)*ky*kdote;
    }
  }
  Kokkos::deep_copy(d_modes_, h_modes_);
}

//acceleration (before normalization) at position (x, y) relative to the box origin
KOKKOS_INLINE_FUNCTION
void turb_raw_accel(const TurbModesView& m, int nm, Real x, Real y, Real& ax, Real& ay){
  ax = 0.0; ay = 0.0;
  for(int n = 0; n < nm; ++n){
    const Real ph = m(n, 0)*x + m(n, 1)*y;
    const Real c = Kokkos::cos(ph), s = Kokkos::sin(ph), w = m(n, 2);
    ax += w*(m(n, 3)*c - m(n, 4)*s); //Re[(Ar + i Ai)(c + i s)]
    ay += w*(m(n, 5)*c - m(n, 6)*s);
  }
}

namespace {
//per active block: slot, x1min, x2min, dx1, dx2 (copied to the device for the kernels)
struct BlockGeo{
  Kokkos::View<int*, MemorySpace> slot;
  Kokkos::View<Real*[4], MemorySpace> geo;
  int n = 0;
};
BlockGeo pack_geometry(Mesh& mesh){
  BlockGeo g;
  g.n = mesh.num_active_blocks();
  g.slot = Kokkos::View<int*, MemorySpace>("turb_slot", g.n);
  g.geo  = Kokkos::View<Real*[4], MemorySpace>("turb_geo", g.n);
  auto hs = Kokkos::create_mirror_view(g.slot);
  auto hg = Kokkos::create_mirror_view(g.geo);
  const auto& blocks = mesh.blocks();
  for(int b = 0; b < g.n; ++b){
    hs(b) = blocks[b]->slot_id;
    hg(b, 0) = blocks[b]->block_size.x1min;
    hg(b, 1) = blocks[b]->block_size.x2min;
    hg(b, 2) = blocks[b]->dx1();
    hg(b, 3) = blocks[b]->dx2();
  }
  Kokkos::deep_copy(g.slot, hs);
  Kokkos::deep_copy(g.geo, hg);
  return g;
}
}

void TurbulenceDriver::begin_step(Mesh& mesh, Real dt){
  //OU update of every amplitude component: A <- f A + sqrt(1 - f^2) P xi
  const Real f = std::exp(-dt/p_.tcorr), sq = std::sqrt(1.0 - f*f);
  const Real zeta = p_.sol_weight;
  const int nm = (int)modes_.size();
  for(int n = 0; n < nm; ++n){
    const Real kx = h_modes_(n, 0), ky = h_modes_(n, 1), k2 = kx*kx + ky*ky;
    for(int part = 0; part < 2; ++part){
      const Real ex = gaussian(), ey = gaussian();
      const Real kdote = (kx*ex + ky*ey)/k2;
      h_modes_(n, 3 + part) = f*h_modes_(n, 3 + part) + sq*(zeta*ex + (1.0 - 2.0*zeta)*kx*kdote);
      h_modes_(n, 5 + part) = f*h_modes_(n, 5 + part) + sq*(zeta*ey + (1.0 - 2.0*zeta)*ky*kdote);
    }
  }
  Kokkos::deep_copy(d_modes_, h_modes_);

  //rms normalization over the whole domain (volume weighted, after removing the volume mean)
  BlockGeo g = pack_geometry(mesh);
  if(g.n == 0) return;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const TurbModesView m = d_modes_;
  const Real x0 = x0_, y0 = y0_;
  Real s_aa = 0.0, s_ax = 0.0, s_ay = 0.0, s_v = 0.0;
  Kokkos::parallel_reduce("TurbRms",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {g.n, ng + nx1, ng + nx2}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& aa, Real& sx, Real& sy, Real& vol){
      const Real dx = g.geo(b, 2), dy = g.geo(b, 3), dV = dx*dy;
      const Real x = g.geo(b, 0) + (i - ng + 0.5)*dx - x0;
      const Real y = g.geo(b, 1) + (j - ng + 0.5)*dy - y0;
      Real ax, ay;
      turb_raw_accel(m, nm, x, y, ax, ay);
      aa += (ax*ax + ay*ay)*dV; sx += ax*dV; sy += ay*dV; vol += dV;
    }, s_aa, s_ax, s_ay, s_v);
  const Real mx = s_ax/s_v, my = s_ay/s_v;
  const Real var = s_aa/s_v - (mx*mx + my*my);
  scale_ = (var > 0.0) ? p_.accel_rms/std::sqrt(var) : 0.0;
}

void TurbulenceDriver::add_forcing(Mesh& mesh, const View4D& U_stage, const View4D& U_out, Real dt) const{
  BlockGeo g = pack_geometry(mesh);
  if(g.n == 0) return;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const TurbModesView m = d_modes_;
  const int nm = (int)modes_.size();
  const Real x0 = x0_, y0 = y0_, scale = scale_;
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {g.n, ng + nx1, ng + nx2});

  //density-weighted mean of the raw field, from the SAME state the force acts on:
  //sum(rho (a - mean) dV) = 0, so the total momentum input vanishes in every stage
  Real s_rx = 0.0, s_ry = 0.0, s_r = 0.0;
  Kokkos::parallel_reduce("TurbMean", policy,
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& rx, Real& ry, Real& rr){
      const int s = g.slot(b);
      const Real dx = g.geo(b, 2), dy = g.geo(b, 3), dV = dx*dy;
      const Real x = g.geo(b, 0) + (i - ng + 0.5)*dx - x0;
      const Real y = g.geo(b, 1) + (j - ng + 0.5)*dy - y0;
      Real ax, ay;
      turb_raw_accel(m, nm, x, y, ax, ay);
      const Real rho = U_stage(s, i, j, IDN);
      rx += rho*ax*dV; ry += rho*ay*dV; rr += rho*dV;
    }, s_rx, s_ry, s_r);
  const Real mx = s_rx/s_r, my = s_ry/s_r;

  Kokkos::parallel_for("TurbForce", policy,
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const int s = g.slot(b);
      const Real dx = g.geo(b, 2), dy = g.geo(b, 3);
      const Real x = g.geo(b, 0) + (i - ng + 0.5)*dx - x0;
      const Real y = g.geo(b, 1) + (j - ng + 0.5)*dy - y0;
      Real ax, ay;
      turb_raw_accel(m, nm, x, y, ax, ay);
      ax = scale*(ax - mx); ay = scale*(ay - my);
      const Real rho = U_stage(s, i, j, IDN);
      U_out(s, i, j, IM1) += dt*rho*ax;
      U_out(s, i, j, IM2) += dt*rho*ay;
      U_out(s, i, j, IEN) += dt*(U_stage(s, i, j, IM1)*ax + U_stage(s, i, j, IM2)*ay); //work rho v.a
    });
}
