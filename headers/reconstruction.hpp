#pragma once

#include "hydro_config.hpp"
#include "slope_limiters.hpp"


/*
 Piecewise-linear (MUSCL) reconstruction of the face states along one direction.

 Options (chosen in the parameter file):
   limiter      minmod (default), vanleer, superbee, mc
   reconstruct  conserved (default): limit rho, rho u, rho v, E
                primitive:           limit rho, u, v, p, then convert the face states back.
                Guarantees positive density and pressure at every face (each face value
                lies between the neighbouring cell values), and keeps u and p uniform
                across a contact discontinuity to round-off.

 This script basically acts as a wrapper for slope_limiters.hpp and lets
 the user specify in the input files which slope limiter they wish to use
*/

//types
enum LimiterType : int { LIM_MINMOD = 0, LIM_VANLEER = 1, LIM_SUPERBEE = 2, LIM_MC = 3 };

//options as structs
struct ReconOptions{
  int  limiter   = LIM_MINMOD;
  bool primitive = false;
};

//just a switch case function for all types
KOKKOS_INLINE_FUNCTION
Vec4 limited_slope(const Vec4& qm1, const Vec4& q0, const Vec4& qp1, Real dk, int limiter){
  switch(limiter){
    case LIM_VANLEER:  return sigma_van_leer(qm1, q0, qp1, dk);
    case LIM_SUPERBEE: return sigma_superbee(qm1, q0, qp1, dk);
    case LIM_MC:       return sigma_mc(qm1, q0, qp1, dk);
    default:           return sigma_minmod(qm1, q0, qp1, dk);
  }
}

//(rho, rho u, rho v, E) -> (rho, u, v, p)
KOKKOS_INLINE_FUNCTION
Vec4 cons_to_prim(const Vec4& q, Real gamma){
  Real rho = q[IDN];
  Real u = q[IM1]/rho, v = q[IM2]/rho;
  Real p = (gamma - 1.0)*(q[IEN] - 0.5*rho*(u*u + v*v));
  return Vec4{rho, u, v, p};
}

//(rho, u, v, p) -> (rho, rho u, rho v, E)
KOKKOS_INLINE_FUNCTION
Vec4 prim_to_cons(const Vec4& w, Real gamma){
  Real rho = w[IDN], u = w[IU], v = w[IV], p = w[IP];
  return Vec4{rho, rho*u, rho*v, p/(gamma - 1.0) + 0.5*rho*(u*u + v*v)};
}

//face states around cell 0 from the 5-cell stencil (m2, m1, 0, p1, p2) along one direction:
//  l_mh, r_mh: left and right state at the face between m1 and 0
//  l_ph, r_ph: left and right state at the face between 0 and p1
KOKKOS_INLINE_FUNCTION
void reconstruct_faces(const Vec4& m2, const Vec4& m1, const Vec4& c0,
                       const Vec4& p1, const Vec4& p2, Real dk,
                       Real gamma, const ReconOptions& opt, 
                       Vec4& l_mh, Vec4& r_mh, Vec4& l_ph, Vec4& r_ph){
  if(!opt.primitive){
    //if not primitive reconstruction then just use the given slope limiter
    Vec4 s_m1 = limited_slope(m2, m1, c0, dk, opt.limiter);
    Vec4 s_0  = limited_slope(m1, c0, p1, dk, opt.limiter);
    Vec4 s_p1 = limited_slope(c0, p1, p2, dk, opt.limiter);
    for(size_t k = 0; k < NVAR; ++k){ //constructing the face states
      l_mh[k] = m1[k] + 0.5*dk*s_m1[k];
      r_mh[k] = c0[k] - 0.5*dk*s_0[k];
      l_ph[k] = c0[k] + 0.5*dk*s_0[k];
      r_ph[k] = p1[k] - 0.5*dk*s_p1[k];
    }
    return;
  }

  //primitive: limit (rho, u, v, p)
  //TVD limiters keep each face value between the neighbouring
  //cell values, so rho and p stay positive at the faces.
  Vec4 wm2 = cons_to_prim(m2, gamma), wm1 = cons_to_prim(m1, gamma), w0 = cons_to_prim(c0, gamma);
  Vec4 wp1 = cons_to_prim(p1, gamma), wp2 = cons_to_prim(p2, gamma);
  Vec4 s_m1 = limited_slope(wm2, wm1, w0, dk, opt.limiter);
  Vec4 s_0  = limited_slope(wm1, w0, wp1, dk, opt.limiter);
  Vec4 s_p1 = limited_slope(w0, wp1, wp2, dk, opt.limiter);
  Vec4 a, b, c, d;
  for(size_t k = 0; k < NVAR; ++k){ //constructing face states
    a[k] = wm1[k] + 0.5*dk*s_m1[k];
    b[k] = w0[k]  - 0.5*dk*s_0[k];
    c[k] = w0[k]  + 0.5*dk*s_0[k];
    d[k] = wp1[k] - 0.5*dk*s_p1[k];
  }
  //since we used primitive vals here we need to convert back to conserved vals
  l_mh = prim_to_cons(a, gamma);
  r_mh = prim_to_cons(b, gamma);
  l_ph = prim_to_cons(c, gamma);
  r_ph = prim_to_cons(d, gamma);
}
