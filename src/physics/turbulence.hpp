#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"

/*
 Few-mode stochastic turbulence driving in 2D (x, y periodic).

 Method: Ornstein-Uhlenbeck forcing in Fourier space 
 (Schmidt et al. 2009; Grete, O'Shea & Beckwith 2018),
 as used by AthenaPK's few-mode driver (Grete et al. 2023, Parthenon). 
 This is an independent 2D implementation; the mode selection follows the same rules:

   modes:      integer wavevectors n = (i, j) with k_low <= |n| <= k_high, spectral shape
               s(k) = (k/k_peak)^2 (2 - (k/k_peak)^2) >= 0, one mode per +-n pair
               (i > 0, or i = 0 and j > 0); k = 2 pi (i/Lx, j/Ly)
   amplitudes: complex 2-vectors A_n, each an OU process with correlation time tcorr,
               driven by noise projected with P = zeta*I + (1 - 2 zeta) k k^T / |k|^2
               (zeta = 1: purely solenoidal, zeta = 0: purely compressive)
   field:      a(x) = sum_n s(|n|) Re[A_n exp(i k.x)], then rescaled to rms = accel_rms and
               shifted so that the density-weighted mean vanishes: the driving adds no net momentum

 Random numbers come from the raw std::mt19937_64 output (whose sequence the C++ standard fixes),
 with our own Gaussian and shuffle: a seed gives the same run with any compiler and standard
 library (std::normal_distribution and std::shuffle do not guarantee that).
*/

struct TurbParams{
  int  k_low  = 1;    //smallest |n| (>= 1)
  Real k_peak = 2.0;  //peak of the spectral shape
  int  k_high = 3;    //largest |n|
  int  nmodes = 0;    //0 = all modes in the shell, otherwise a random subset of this size
  Real tcorr  = 1.0;  //OU correlation time
  Real accel_rms  = 1.0;  //rms of the acceleration field
  Real sol_weight = 1.0;  //zeta: 1 solenoidal, 0 compressive
  std::uint64_t seed = 1;
};

//all modes (i, j) of the shell, in the order of the loops (i = 0..k_high, j = -k_high..k_high)
std::vector<std::array<int, 2>> TurbulenceModeList(int k_low, Real k_peak, int k_high);

//per mode: kx, ky, shape weight, Re A_x, Im A_x, Re A_y, Im A_y
using TurbModesView = Kokkos::View<Real*[7], Kokkos::LayoutRight, MemorySpace>;

class TurbulenceDriver{
  private:
    Real uniform01();  //from the raw generator output: same sequence everywhere
    Real gaussian();   //Box-Muller

    TurbParams p_;
    Real x0_, y0_;
    std::vector<std::array<int, 2>> modes_;
    std::mt19937_64 rng_;
    bool have_spare_ = false;
    Real spare_ = 0.0;
    TurbModesView d_modes_;
    TurbModesView::host_mirror_type h_modes_;
    Real scale_ = 0.0;  //rms normalization factor, set in begin_step

  public:
    //box [x0, x0 + Lx] x [y0, y0 + Ly]; "Parameter error" for invalid settings
    TurbulenceDriver(const TurbParams& p, Real x0, Real y0, Real Lx, Real Ly);

    //once per step, before the integrator: OU update of the amplitudes and the rms normalization
    //(geometric: the volume-weighted rms of the field over the domain)
    void begin_step(Mesh& mesh, Real dt);

    //smooth source term (every stage): rho*a into the momentum, rho*v.a into the energy, with the
    //density-weighted mean of a computed from U_stage itself, so the net momentum input is zero
    void add_forcing(Mesh& mesh, const View4D& U_stage, const View4D& U_out, Real dt) const;

    const std::vector<std::array<int, 2>>& modes() const { return modes_; }

};
