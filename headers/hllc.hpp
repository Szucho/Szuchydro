#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "hydro_config.hpp"
#include "vector_ops.hpp"

/*
 HLLC method for 2D Euler HD equation header file
 Bertalan Szuchovszky 26.02.2026
 
 Modified on 21.09.2026:
  - Ported state vectors to the zero-allocation Vec4 struct (hydro_config.hpp).
  - Added KOKKOS_INLINE_FUNCTION decorators to facilitate execution on host and device backends.
  - Consolidated header includes for Modern CMake / Kokkos integration.

 state vector Q = [rho, rho*u, rho*v, rho*e_tot] 
 with e_tot = e_th + 0.5*|v|^2 & e_th = 1/(gamma-1) k_BT/mu m_p thermal energy

 We only need the hllc flux from the HLLC.cpp file
 FluxhllcX : numerical flux at an interface in x direction
 FluxhllcY : numerical flux at an interface in y direction

 Input:
  QL, QR: left and right state vectors at the interface!!! USE SLOPE LIMITERS TO GET THEM!
  gamma: (f+2)/f adiabatic index
 Returns:
  HLLC flux vectors in x direction or y direction


  For details read Toro - Riemann Solvers and Numerical Methods for Fluid Dynamics (3rd ed)
  I implemented chapter 10 using my matrix.h header for numpy like arrays.
  This code serves to calculate the hllc flux with given Q state vector and gamma adiabatic index.
  The Toro book uses SK instead of lambdas, but to stay consistent with the FSM lecture notation
  I will use lambda_{m}, lambda_{0} and lambda_{+} instead.

*/

struct Primitive_Vals{
  double rho, u, v, p, cs;
};

KOKKOS_INLINE_FUNCTION
static Primitive_Vals QtoPrim(const Vec4& Q, double gamma){
  Primitive_Vals q;
  q.rho = Q[0];
  q.u = Q[1]/q.rho; //rho u / rho
  q.v = Q[2]/q.rho; //rho v / rho

  //Q[3] = rho e_tot = rho(e_th + 0.5(u^2 + v^2)), e_th = 1/(gamma-1)k_BT/mu m_p
  //p = rho k_BT/mu m_p ideal gas law -> p = (gamma-1)*e_th = (gamma-1)rho(e_tot-0.5(u^2+v^2))
  q.p = (gamma - 1.0)*(Q[3] - q.rho*0.5*(q.u*q.u + q.v*q.v));

  if(q.rho < 0.0 || q.p < 0.0){
    //exceptions are host-only: still catchable in main on the Serial backend, abort on device
    KOKKOS_IF_ON_HOST((throw std::invalid_argument("Density or pressure is negative");))
    KOKKOS_IF_ON_DEVICE((Kokkos::abort("Density or pressure is negative");))
  }
  q.cs = Kokkos::sqrt(gamma*q.p/q.rho);

  return q;
}

//flux in the x direction - Toro chapter 10
KOKKOS_INLINE_FUNCTION
static Vec4 xFlux(const Vec4& Q, double gamma){
  Primitive_Vals q = QtoPrim(Q, gamma);
  double e_tot = Q[3]/q.rho;
  double h_tot = e_tot + q.p/q.rho;
  //Fx = (rho u, rho u^2 + p, rho u v, rho u h_tot)
  return Vec4{
    q.rho*q.u,
    q.rho*q.u*q.u + q.p,
    q.rho*q.v*q.u,
    q.rho*q.u*h_tot
  };
}

struct lambdas {
  double lm, l0, lp; //lambda_{-}, lambda_{0}, lambda_{+} eigenvals of the Jacobi matrix
};

KOKKOS_INLINE_FUNCTION
static lambdas WavecX(const Primitive_Vals& L, const Primitive_Vals& R){
  //Toro 10.48
  double SL = Kokkos::min(L.u - L.cs, R.u - R.cs); //lambda_{-} = u-cs but let it be the min of these
  double SR = Kokkos::max(L.u + L.cs, R.u + R.cs); //lambda_{+} = u+cs and the max of these

  //Toro eq. 10.36
  double Sstar = (R.p - L.p + L.rho*L.u*(SL - L.u) - R.rho*R.u*(SR - R.u))/(L.rho*(SL - L.u) - R.rho*(SR - R.u));
  return {SL, Sstar, SR};
}

KOKKOS_INLINE_FUNCTION
static Vec4 Qtilde(const Vec4& Q, double gamma, double SK, double Sstar){
  //Q is Q_K with K=L || K=R => q is also q_k
  Primitive_Vals q = QtoPrim(Q, gamma);
  double mul = q.rho*(SK - q.u)/(SK - Sstar);
  double E = Q[3];
  //Toro 10.39
  return Vec4{
    mul,
    mul*Sstar,
    mul*q.v,
    mul*(E/q.rho + (Sstar - q.u)*(Sstar + q.p/(q.rho*(SK - q.u))))
  };
}

KOKKOS_INLINE_FUNCTION
Vec4 FluxhllcX(const Vec4& QL, const Vec4& QR, double gamma){
  //left and right primitive vals
  Primitive_Vals L = QtoPrim(QL, gamma);
  Primitive_Vals R = QtoPrim(QR, gamma);

  lambdas w = WavecX(L, R);

  //Toro 10.26
  if(0.0 <= w.lm){
    return xFlux(QL, gamma);
  }

  if(w.lp <= 0.0){
    return xFlux(QR, gamma);
  }

  if(w.lm <= 0.0 && 0.0 <= w.l0){
    Vec4 QtL = Qtilde(QL, gamma, w.lm, w.l0); //Toro 10.27
    return xFlux(QL, gamma) + w.lm*(QtL - QL);
  }

  Vec4 QtR = Qtilde(QR, gamma, w.lp, w.l0); //Toro 10.29
  return xFlux(QR, gamma) + w.lp*(QtR - QR);
}

//trick to get the flux in y direction instead of writing the whole thing again
KOKKOS_INLINE_FUNCTION
static Vec4 rotateQxy(const Vec4& Q){
  //[rho, rho*u, rho*v, E]  ->  [rho, rho*v, rho*u, E]
  return Vec4{Q[0], Q[2], Q[1], Q[3]};
}

KOKKOS_INLINE_FUNCTION
Vec4 FluxhllcY(const Vec4& QL, const Vec4& QR, double gamma){
  //rotate so that y becomes the normal direction
  Vec4 QL_rot = rotateQxy(QL);
  Vec4 QR_rot = rotateQxy(QR);

  //solve as an x direction flux but now with rotated Q
  Vec4 Frot = FluxhllcX(QL_rot, QR_rot, gamma);

  //rotate flux back:  G = [rho*v, rho*u*v, rho*v^2+p, (E+p)*v]
  return Vec4{Frot[0], Frot[2], Frot[1], Frot[3]};
}
