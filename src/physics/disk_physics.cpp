#include "physics/disk_physics.hpp"
#include <stdexcept>

//tabulate S(eta) = -int_0^eta eta'/h(eta')^2 deta'  (trapezoid, fine grid)
DiskModel BuildDiskModel(const DiskParams& p, Real rho0, Real eta_max){
  DiskModel m;
  m.p = p;
  m.rho0 = rho0;
  m.ntab = 8192;
  m.deta = eta_max/(m.ntab - 1);
  m.S_table = View1D("disk_S_table", m.ntab);
  auto h_tab = Kokkos::create_mirror_view(m.S_table);

  auto integrand = [&](Real e){
    Real h = disk_h(e, p.h_disk, p.h_wind, p.eta_tr);
    return -e/(h*h);
  };
  h_tab(0) = 0.0;
  for(int k = 1; k < m.ntab; ++k){
    Real e0 = (k - 1)*m.deta, e1 = k*m.deta;
    h_tab(k) = h_tab(k - 1) + 0.5*m.deta*(integrand(e0) + integrand(e1));
  }
  Kokkos::deep_copy(m.S_table, h_tab);
  return m;
}

//fills every block (ghosts included) with the hydrostatic disk + corona
void InitDiskRz(Mesh& mesh, const DiskModel& m, Real gamma){
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;
  const View1D tab = m.S_table;
  const Real deta = m.deta; const int ntab = m.ntab;
  const Real hd = m.p.h_disk, hw = m.p.h_wind, etr = m.p.eta_tr, qD = m.p.q_D;
  const Real rho0 = m.rho0, rfl = m.p.rho_floor, pfl = m.p.p_floor;

  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    if(b.block_size.x1min - nghost*b.dx1() <= 0.0)
      throw std::runtime_error("disk_Rz needs xmin > nghost*dx (x1 is cylindrical R > 0)");
    //plain scalars only inside the kernel (GPU-safe)
    const int slot = b.slot_id;
    const Real x0 = b.block_size.x1min, y0 = b.block_size.x2min;
    const Real dx = b.dx1(), dy = b.dx2();
    const int is = b.is, js = b.js;
    const int ni = b.block_size.nx1 + 2*nghost, nj = b.block_size.nx2 + 2*nghost;

    Kokkos::parallel_for("InitDiskRz", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ni, nj}),
      KOKKOS_LAMBDA(int i, int j){
        Real R = x0 + (i - is + 0.5)*dx;
        Real z = y0 + (j - js + 0.5)*dy;
        Real aeta = std::abs(z/R);
        Real h = disk_h(aeta, hd, hw, etr);
        Real S = disk_S(tab, deta, ntab, aeta);
        Real P = rho0*std::pow(R, -qD - 1.0)*std::exp(S);
        Real T = h*h/R;
        Real rho = P/T;
        Real rmin = rfl*rho0*std::pow(R, -qD);
        if(rho < rmin){ rho = rmin; P = rho*T; }
        if(P < pfl) P = pfl;
        data(slot, i, j, IDN) = rho;
        data(slot, i, j, IM1) = 0.0;
        data(slot, i, j, IM2) = 0.0;
        data(slot, i, j, IEN) = P/(gamma - 1.0);
      });
  }
  Kokkos::fence();
}

StirringForce::StirringForce(const DiskParams& p)
  : n_(p.stir_nmodes), tcorr_(p.stir_tcorr), rng_(p.stir_seed){
  if(n_ < 1) n_ = 1;
  modes_ = StirModesView("stir_modes", n_);
  h_modes_ = Kokkos::create_mirror_view(modes_);
  //random wavevectors with |k| in [2pi/lmax, 2pi/lmin], random direction
  std::uniform_real_distribution<Real> uni(0.0, 1.0);
  const Real kmin = 2.0*M_PI/p.stir_lmax, kmax = 2.0*M_PI/p.stir_lmin;
  for(int m = 0; m < n_; ++m){
    Real k = kmin + (kmax - kmin)*uni(rng_);
    Real th = 2.0*M_PI*uni(rng_);
    h_modes_(m, 0) = k*std::cos(th);
    h_modes_(m, 1) = k*std::sin(th);
    h_modes_(m, 2) = gauss_(rng_);  //start in the stationary OU distribution
    h_modes_(m, 3) = gauss_(rng_);
  }
  Kokkos::deep_copy(modes_, h_modes_);
}

void StirringForce::Advance(Real dt){
  //exact Ornstein-Uhlenbeck update, unit stationary variance
  Real f = std::exp(-dt/tcorr_);
  Real s = std::sqrt(1.0 - f*f);
  for(int m = 0; m < n_; ++m){
    h_modes_(m, 2) = f*h_modes_(m, 2) + s*gauss_(rng_);
    h_modes_(m, 3) = f*h_modes_(m, 3) + s*gauss_(rng_);
  }
  Kokkos::deep_copy(modes_, h_modes_);
}

void AddDiskSourceTerms(Mesh& mesh, const DiskModel& m, const StirringForce& stir,
                        Real dt, Real gamma){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  View4D data = mesh.get_data_view();
  const int nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int ng = mesh.inputs().nghost;

  //per-block geometry: slot, x1min, x2min, dx1, dx2
  Kokkos::View<Real*[4], Kokkos::HostSpace> h_geo("h_geo", n_active);
  Kokkos::View<int*, Kokkos::HostSpace> h_slot("h_slot", n_active);
  const auto& blocks = mesh.blocks();
  for(int b = 0; b < n_active; ++b){
    h_slot(b) = blocks[b]->slot_id;
    h_geo(b, 0) = blocks[b]->block_size.x1min;
    h_geo(b, 1) = blocks[b]->block_size.x2min;
    h_geo(b, 2) = blocks[b]->dx1();
    h_geo(b, 3) = blocks[b]->dx2();
  }
  Kokkos::View<Real*[4]> geo("geo", n_active);
  Kokkos::View<int*> slot("slot", n_active);
  Kokkos::deep_copy(geo, h_geo);
  Kokkos::deep_copy(slot, h_slot);

  const View1D tab = m.S_table;
  const Real deta = m.deta; const int ntab = m.ntab;
  const DiskParams p = m.p;
  const Real rho0 = m.rho0;
  const StirModesView modes = stir.modes();
  const int nm = stir.nmodes();
  const Real norm = 1.0/std::sqrt((Real)nm);
  const Real sa = std::sin(p.wind_angle*M_PI/180.0), ca = std::cos(p.wind_angle*M_PI/180.0);

  Kokkos::parallel_for("DiskSources",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ng + nx1, ng + nx2}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const int s = slot(b);
      const Real R = geo(b, 0) + (i - ng + 0.5)*geo(b, 2);
      const Real z = geo(b, 1) + (j - ng + 0.5)*geo(b, 3);
      const Real eta = z/R, aeta = std::abs(eta);
      const Real Om = std::pow(R, -1.5);
      const Real h  = disk_h(aeta, p.h_disk, p.h_wind, p.eta_tr);
      const Real T0 = h*h/R;                    //target temperature (P/rho)
      const Real ww = disk_wind_weight(aeta, p.h_disk, p.eta_tr); //1 in corona, 0 in disk
      const Real wd = 1.0 - ww;
      const Real S  = disk_S(tab, deta, ntab, aeta);
      const Real rho_i = rho0*std::pow(R, -p.q_D)*std::exp(S)/(h*h);

      //current primitives
      Real rho = data(s, i, j, IDN);
      Real u = data(s, i, j, IM1)/rho;
      Real v = data(s, i, j, IM2)/rho;
      Real eint = data(s, i, j, IEN) - 0.5*rho*(u*u + v*v);

      //corona replenishment (velocity kept, internal energy per mass kept)
      {
        Real fr = ww*(1.0 - std::exp(-dt*Om/p.corona_tau));
        Real rho_new = rho + (rho_i - rho)*fr;
        eint *= rho_new/rho;
        rho = rho_new;
      }
      //equilibrium force grad(P_init)/rho_init
      {
        Real fR = (eta*eta - (p.q_D + 1.0)*h*h)/(R*R);
        Real fz = -eta/(R*R);
        u += dt*fR;
        v += dt*fz;
      }
      //stirring: solenoidal in (xi = ln R, eta) space, disk band only
      if(wd > 1e-3){
        const Real xi = std::log(R);
        Real axi = 0.0, aet = 0.0;
        for(int k = 0; k < nm; ++k){
          Real kx = modes(k, 0), ke = modes(k, 1);
          Real kk = std::sqrt(kx*kx + ke*ke);
          Real ph = kx*xi + ke*eta;
          Real d = (-modes(k, 2)*std::sin(ph) + modes(k, 3)*std::cos(ph))/kk;
          axi +=  ke*d;   // dpsi/deta
          aet += -kx*d;   //-dpsi/dxi
        }
        //OU forcing a with drag time tau_d gives v_rms ~ a*sqrt(tau_c*tau_d);
        //choose a so v_rms ~ stir_amp*c_s
        Real cs_d = p.h_disk/std::sqrt(R);
        Real acc = p.stir_amp*cs_d*std::sqrt(Om/(p.stir_tcorr*p.stir_damp))*wd*norm;
        u += dt*acc*axi;
        v += dt*acc*aet;
        //large-scale drag (disk band only)
        Real fd = wd*(1.0 - std::exp(-dt*Om/p.stir_damp));
        u -= u*fd;
        v -= v*fd;
      }
      //4) wind: relax toward tilted outflow in the corona
      {
        Real cs = h/std::sqrt(R);
        Real sgn = (z >= 0.0) ? 1.0 : -1.0;
        Real ut = p.wind_mach*cs*sa;
        Real vt = p.wind_mach*cs*ca*sgn;
        Real fr = ww*(1.0 - std::exp(-dt*Om/p.wind_tau));
        u += (ut - u)*fr;
        v += (vt - v)*fr;
      }
      //5) beta cooling toward T0
      {
        Real e_target = rho*T0/(gamma - 1.0);
        Real fr = 1.0 - std::exp(-dt*Om/p.beta_cool);
        eint += (e_target - eint)*fr;
      }
      //6) floors
      Real rmin = p.rho_floor*rho0*std::pow(R, -p.q_D);
      if(rho < rmin) rho = rmin;
      Real emin = p.p_floor/(gamma - 1.0);
      if(eint < emin) eint = emin;

      data(s, i, j, IDN) = rho;
      data(s, i, j, IM1) = rho*u;
      data(s, i, j, IM2) = rho*v;
      data(s, i, j, IEN) = eint + 0.5*rho*(u*u + v*v);
    });
}

//Paper-style disk boundaries (Huang & Bai Sec. II.2): in ghost cells on physical
//walls, density and pressure are reset to the initial profile, velocities are kept
//from the regular BC (use Outflow so inflow is clamped). Call AFTER
//ApplyPhysicalBoundaryConditions.
void ApplyDiskBoundaries(Mesh& mesh, const DiskModel& m, Real gamma){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  View4D data = mesh.get_data_view();
  const int nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int ng = mesh.inputs().nghost;

  Kokkos::View<Real*[4], Kokkos::HostSpace> h_geo("h_geo_bc", n_active);
  Kokkos::View<int*[5], Kokkos::HostSpace> h_info("h_info_bc", n_active); //slot, phys XM,XP,YM,YP
  const auto& blocks = mesh.blocks();
  for(int b = 0; b < n_active; ++b){
    h_info(b, 0) = blocks[b]->slot_id;
    for(int f = 0; f < 4; ++f) h_info(b, 1 + f) = blocks[b]->is_physical_boundary[f] ? 1 : 0;
    h_geo(b, 0) = blocks[b]->block_size.x1min;
    h_geo(b, 1) = blocks[b]->block_size.x2min;
    h_geo(b, 2) = blocks[b]->dx1();
    h_geo(b, 3) = blocks[b]->dx2();
  }
  Kokkos::View<Real*[4]> geo("geo_bc", n_active);
  Kokkos::View<int*[5]> info("info_bc", n_active);
  Kokkos::deep_copy(geo, h_geo);
  Kokkos::deep_copy(info, h_info);

  const View1D tab = m.S_table;
  const Real deta = m.deta; const int ntab = m.ntab;
  const DiskParams p = m.p;
  const Real rho0 = m.rho0;

  Kokkos::parallel_for("DiskBoundaries",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, nx1 + 2*ng, nx2 + 2*ng}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const bool gx_lo = (i < ng)        && info(b, 1);
      const bool gx_hi = (i >= ng + nx1) && info(b, 2);
      const bool gy_lo = (j < ng)        && info(b, 3);
      const bool gy_hi = (j >= ng + nx2) && info(b, 4);
      if(!(gx_lo || gx_hi || gy_lo || gy_hi)) return;
      const int s = info(b, 0);
      const Real R = geo(b, 0) + (i - ng + 0.5)*geo(b, 2);
      const Real z = geo(b, 1) + (j - ng + 0.5)*geo(b, 3);
      const Real aeta = std::abs(z/R);
      const Real h = disk_h(aeta, p.h_disk, p.h_wind, p.eta_tr);
      const Real S = disk_S(tab, deta, ntab, aeta);
      Real P = rho0*std::pow(R, -p.q_D - 1.0)*std::exp(S);
      Real rho = P*R/(h*h);
      const Real rmin = p.rho_floor*rho0*std::pow(R, -p.q_D);
      if(rho < rmin){ rho = rmin; P = rho*h*h/R; }
      //velocities from the regular BC
      const Real rho_old = data(s, i, j, IDN);
      const Real u = data(s, i, j, IM1)/rho_old, v = data(s, i, j, IM2)/rho_old;
      data(s, i, j, IDN) = rho;
      data(s, i, j, IM1) = rho*u;
      data(s, i, j, IM2) = rho*v;
      data(s, i, j, IEN) = P/(gamma - 1.0) + 0.5*rho*(u*u + v*v);
    });
}
