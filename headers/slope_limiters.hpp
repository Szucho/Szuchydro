#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "hydro_config.hpp"
#include "vector_ops.hpp"

/*
 Slope limiter header file
 Bertalan Szuchovszky 26.06.2026

 Modified on 21.09.2026:
  - Renamed file to slope_limiters.hpp for project consistency.
  - Ported state vectors to Kokkos::Array<double, 4> (Vec4).
  - Added KOKKOS_INLINE_FUNCTION decorators for CPU/GPU parallel execution.

 state vector Q = [rho, rho*u, rho*v, rho*e_tot] 
 with e_tot = e_th + 0.5*|v|^2 & e_th = 1/(gamma-1) k_BT/mu m_p thermal energy

 Funcs: sigma_superbee, sigma_minmod, sigma_van_leer

 Input:
  Qim1, Qi, Qip1: Q_{i-1}, Q_{i}, Q_{i+1} state vectors 
  dk: either dx or dy depending on direction
 Output:
  sigma_k slope vector
*/

KOKKOS_INLINE_FUNCTION
Real superbee_one(Real qim1, Real qi, Real qip1, Real dx);

KOKKOS_INLINE_FUNCTION
Real minmod_one(Real qim1, Real qi, Real qip1, Real dk);

KOKKOS_INLINE_FUNCTION
Real van_leer_one(Real qim1, Real qi, Real qip1, Real dk);

KOKKOS_INLINE_FUNCTION
Vec4 sigma_superbee(const Vec4& Qim1, const Vec4& Qi, const Vec4& Qip1, Real dk);

KOKKOS_INLINE_FUNCTION
Vec4 sigma_minmod(const Vec4& Qim1, const Vec4& Qi, const Vec4& Qip1, Real dk);

KOKKOS_INLINE_FUNCTION
Vec4 sigma_van_leer(const Vec4& Qim1, const Vec4& Qi, const Vec4& Qip1, Real dk);

/*
 * HAD TO THROW THE CPP FILE CONTENT IN HERE FOR CMAKE
 Slope limiter script
 Bertalan Szuchovszky 26.02.2026

 Based on FSM lecture notes, surely this can be found online as well
 The main function is sigma superbee:
    -> calculates slope based on Q_{i-1}, Q_{i}, Q_{i+1} and dx or dy
    Slopes available:
        - superbee
        - minmod
        - Van Leer
    -> will be used to estimate Q_L and Q_R values at cell interfaces
 Every other function is a helper function
 

 Modified from 20.04.2026 - 07.05.2026:
 - added Van Leer, minmod slope limiters

 Modified on 21.09.2026:
 - Replaced dynamic Vector with Kokkos::Array<double, 4> (Vec4).
 - Added KOKKOS_INLINE_FUNCTION for device compilation.
*/

using namespace VecOps;

//minmod(a,b) = b if a*b>0, |a| > |b|
//             = a if a*b>0, |a| <= |b|
//             = 0 if a*b<0
KOKKOS_INLINE_FUNCTION
static Real minmod(Real a, Real b){
  if(a*b < 0.0) return 0.0;
  else if (std::abs(a) > std::abs(b)) return b;
  else return a;
}

//minmod(a,b) = a if a*b>0, |a| > |b|
//             = b if a*b>0, |a| <= |b|
//             = 0 if a*b<0
KOKKOS_INLINE_FUNCTION
static Real maxmod(Real a, Real b){
  if(a*b < 0.0) return 0.0;
  else if (std::abs(a) > std::abs(b)) return a;
  else return b;
}

//superbee: maxmod(s1,s2) with s1,s2 being minmod slopes with the proper arguments
KOKKOS_INLINE_FUNCTION
Real superbee_one(Real qim1, Real qi, Real qip1, Real dx){ //dx or dy depending on direction
  Real s1 = minmod((qi - qim1)/dx, 2.0*(qip1 - qi)/dx);
  Real s2 = minmod(2.0*(qi - qim1)/dx, (qip1 - qi)/dx);
  return maxmod(s1, s2);
}

//Q = [rho, rho u, rho v, rho e_tot], i index means i-th gridpoint
//vectorial version, calculate superbee slope for every element of state vector
KOKKOS_INLINE_FUNCTION
Vec4 sigma_superbee(const Vec4& Qim1, const Vec4& Qi, const Vec4& Qip1, Real dk){
  Real s0 = superbee_one(Qim1[0], Qi[0], Qip1[0], dk); //k = x or y depending on direction
  Real s1 = superbee_one(Qim1[1], Qi[1], Qip1[1], dk); //in practice:
  Real s2 = superbee_one(Qim1[2], Qi[2], Qip1[2], dk); //sigma_x = sigma_superbee(Qim1,Qi,Qip1,dx)
  Real s3 = superbee_one(Qim1[3], Qi[3], Qip1[3], dk); //sigma_y = sigma_superbee(Qim1,Qi,Qip1,dy)
  return Vec4{s0, s1, s2, s3};
}

KOKKOS_INLINE_FUNCTION
Real minmod_one(double qim1, double qi, double qip1, Real dk){
  Real sL = (qi - qim1)/dk;
  Real sR = (qip1 - qi)/dk;
  return minmod(sL, sR); //reuses existing minmod(a,b)
}

KOKKOS_INLINE_FUNCTION
Vec4 sigma_minmod(const Vec4& Qim1, const Vec4& Qi, const Vec4& Qip1, Real dk){
  Vec4 sigma;
  for(size_t i = 0; i < 4; i++){
    sigma[i] = minmod_one(Qim1[i], Qi[i], Qip1[i], dk);
  }
  return sigma;
}

KOKKOS_INLINE_FUNCTION
Real van_leer_one(Real qim1, Real qi, Real qip1, Real dk){
  Real sL = (qi - qim1)/dk;
  Real sR = (qip1 - qi)/dk;

  if(sL*sR <= 0.0){
    return 0.0;
  }
  return (2.0*sL*sR)/(sL + sR);
}

KOKKOS_INLINE_FUNCTION
Vec4 sigma_van_leer(const Vec4& Qim1, const Vec4& Qi, const Vec4& Qip1, Real dk){
  Vec4 sigma;
  for(size_t i = 0; i < 4; i++){
    sigma[i] = van_leer_one(Qim1[i], Qi[i], Qip1[i], dk);
  }
  return sigma;
}
