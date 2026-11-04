#pragma once

#include <vector>
#include <random>
#include <cmath>
#include "hydro_config.hpp"
#include "mesh.hpp"

/*
 * Stylized protoplanetary-disk physics for an R-z (meridional) slice.
 * x1 = R (cylindrical radius, must be > 0), x2 = z.
 *
 * Self-similar disk following Huang & Bai (2026) Sec. II.2 in spirit:
 *   eta = z/R, h(eta) = H/R switches from h_disk to h_wind at |eta| = eta_tr (tanh, Eq. 13)
 *   T_init = h(eta)^2 / R                     (G = M = 1)
 *   P_init = rho0 * R^(-qD-1) * exp(S(eta))   with dS/deta = -eta/h^2  (vertical hydrostatics)
 *   rho_init = P_init / T_init
 *
 * Source terms (operator split, applied after the hydro step):
 *   1) corona density replenishment toward rho_init
 *   2) equilibrium force f = grad(P_init)/rho_init  (stands in for gravity + rotation)
 *   3) solenoidal stochastic stirring inside the disk band (OU-correlated Fourier modes)
 *   4) wind: velocity relaxed toward a tilted outflow in the corona
 *   5) beta-cooling toward T_init
 *   6) density / pressure floors
 */

struct DiskParams{
  Real h_disk = 0.1, h_wind = 0.5, eta_tr = 0.35, q_D = 2.25;
  Real rho_floor = 1e-8;   //relative to R^(-qD)*rho0
  Real p_floor   = 1e-12;
  Real beta_cool = 0.01;   //cooling time in units of 1/Omega_K
  //stirring
  int  stir_nmodes = 24;
  Real stir_amp    = 0.2;  //target rms Mach number of the stirred disk
  Real stir_tcorr  = 2.0;  //forcing correlation time in code units
  Real stir_damp   = 10.0; //large-scale drag time in units of 1/Omega_K (saturates the stirring)
  Real stir_lmin   = 0.15; //wavelength range in (ln R, z/R) space, i.e. in units of R
  Real stir_lmax   = 0.40;
  unsigned stir_seed = 42;
  //wind
  Real wind_mach  = 1.5;   //target speed in units of local c_s
  Real wind_angle = 30.0;  //degrees, tilt away from vertical toward +R
  Real wind_tau   = 1.0;   //relaxation time in units of 1/Omega_K
  Real corona_tau = 5.0;   //density replenishment time in units of 1/Omega_K
};

//stirring mode table (nmodes, 4): k_xi, k_eta, a, b
using StirModesView = Kokkos::View<Real*[4], Kokkos::LayoutRight, MemorySpace>;

//everything the device kernels need
struct DiskModel{
  DiskParams p;
  Real rho0 = 1.0;
  View1D S_table;  //S(eta) on a uniform grid in |eta|
  Real deta = 0.0;
  int ntab = 0;
};

//device helpers
KOKKOS_INLINE_FUNCTION
Real disk_wind_weight(Real aeta, Real h_disk, Real eta_tr){
  return 0.5*(std::tanh(2.0*(aeta - eta_tr)/h_disk) + 1.0);
}

KOKKOS_INLINE_FUNCTION
Real disk_h(Real aeta, Real h_disk, Real h_wind, Real eta_tr){
  return h_disk + (h_wind - h_disk)*disk_wind_weight(aeta, h_disk, eta_tr);
}

KOKKOS_INLINE_FUNCTION
Real disk_S(const View1D& tab, Real deta, int ntab, Real aeta){
  Real s = aeta/deta;
  int k = (int)s;
  if(k >= ntab - 1) return tab(ntab - 1);
  Real w = s - k;
  return (1.0 - w)*tab(k) + w*tab(k + 1);
}

//host API
DiskModel BuildDiskModel(const DiskParams& p, Real rho0, Real eta_max);
void InitDiskRz(Mesh& mesh, const DiskModel& m, Real gamma);

class StirringForce{
public:
  StirringForce(const DiskParams& p);
  void Advance(Real dt);  //OU update of mode amplitudes, then copies to device
  StirModesView modes() const { return modes_; }
  int nmodes() const { return n_; }
private:
  int n_;
  Real tcorr_;
  std::mt19937 rng_;
  std::normal_distribution<Real> gauss_{0.0, 1.0};
  StirModesView modes_;
  StirModesView::host_mirror_type h_modes_;
};

void ApplyDiskBoundaries(Mesh& mesh, const DiskModel& m, Real gamma);
void AddDiskSourceTerms(Mesh& mesh, const DiskModel& m, const StirringForce& stir,
                        Real dt, Real gamma);
