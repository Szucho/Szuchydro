#pragma once

#include "core/hydro_config.hpp"
#include "hydro/slope_limiters.hpp"


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

//Left and right state at ONE face, the face between cells m1 and c0, from the 4-cell stencil
//(m2, m1 | c0, p1) along one direction:
//  l: from cell m1, extrapolated to its right edge with the slope over (m2, m1, c0)
//  r: from cell c0, extrapolated to its left edge with the slope over (m1, c0, p1)
//Each face is computed once (by ComputeFluxes in solver/rhs.cpp), with exactly the
//arithmetic the old per-cell version used, so results are bit-for-bit unchanged.
KOKKOS_INLINE_FUNCTION
void reconstruct_face(const Vec4& m2, const Vec4& m1, const Vec4& c0, const Vec4& p1,
                      Real dk, Real gamma, const ReconOptions& opt, Vec4& l, Vec4& r){
  if(!opt.primitive){
    //conserved reconstruction: limit (rho, rho u, rho v, E) directly
    Vec4 s_m1 = limited_slope(m2, m1, c0, dk, opt.limiter);
    Vec4 s_0  = limited_slope(m1, c0, p1, dk, opt.limiter);
    for(size_t k = 0; k < NVAR; ++k){ //constructing the face states
      l[k] = m1[k] + 0.5*dk*s_m1[k];
      r[k] = c0[k] - 0.5*dk*s_0[k];
    }
    return;
  }

  //primitive: limit (rho, u, v, p)
  //TVD limiters keep each face value between the neighbouring
  //cell values, so rho and p stay positive at the faces.
  Vec4 wm2 = cons_to_prim(m2, gamma), wm1 = cons_to_prim(m1, gamma);
  Vec4 w0 = cons_to_prim(c0, gamma), wp1 = cons_to_prim(p1, gamma);
  Vec4 s_m1 = limited_slope(wm2, wm1, w0, dk, opt.limiter);
  Vec4 s_0  = limited_slope(wm1, w0, wp1, dk, opt.limiter);
  Vec4 a, b;
  for(size_t k = 0; k < NVAR; ++k){ //constructing face states
    a[k] = wm1[k] + 0.5*dk*s_m1[k];
    b[k] = w0[k]  - 0.5*dk*s_0[k];
  }
  //since we used primitive vals here we need to convert back to conserved vals
  l = prim_to_cons(a, gamma);
  r = prim_to_cons(b, gamma);
}

//Well-balanced variant of reconstruct_face for an external potential phi (Käppeli & Mishra 2016,
//linear hydrostatic extrapolation). rho, u, v are reconstructed as in the primitive branch: the
//pressure is split in each cell into a hydrostatic part p0, extrapolated with phi, and a
//perturbation p1 = p - p0, and only p1 is limited:
//  in cell i:     p0(x) = p_i - rho_i (phi(x) - phi_i),   phi at a face = mean of the two cells
//  neighbours:    p0_{i+-1} = p_i -+ (rho_i + rho_{i+-1})/2 (phi_{i+-1} - phi_i)
//                 p1_{i+-1} = p_{i+-1} - p0_{i+-1},  p1_i = 0
//In a discrete hydrostatic equilibrium, p_{i+1} - p_i = -(rho_i + rho_{i+1})/2 (phi_{i+1} - phi_i),
//all p1 vanish, both sides of every face get the same pressure, and the flux difference cancels
//the potential source -rho_i (phi_{i+1} - phi_{i-1})/(2 dx) to machine precision. This holds for
//ANY temperature profile in a potential that varies along the reconstruction direction.
KOKKOS_INLINE_FUNCTION
void reconstruct_face_wb(const Vec4& m2, const Vec4& m1, const Vec4& c0, const Vec4& p1,
                         Real phi_m2, Real phi_m1, Real phi_c0, Real phi_p1,
                         Real dk, Real gamma, int limiter, Vec4& l, Vec4& r){
  Vec4 wm2 = cons_to_prim(m2, gamma), wm1 = cons_to_prim(m1, gamma);
  Vec4 w0  = cons_to_prim(c0, gamma), wp1 = cons_to_prim(p1, gamma);
  //rho, u, v: ordinary limited reconstruction (the pressure component is replaced below)
  Vec4 s_m1 = limited_slope(wm2, wm1, w0, dk, limiter);
  Vec4 s_0  = limited_slope(wm1, w0, wp1, dk, limiter);
  Vec4 a, b;
  for(size_t k = 0; k < NVAR; ++k){
    a[k] = wm1[k] + 0.5*dk*s_m1[k];
    b[k] = w0[k]  - 0.5*dk*s_0[k];
  }
  //left state (cell m1, extrapolated to its right edge)
  {
    const Real rho = wm1[IDN], p = wm1[IP];
    const Real p0_face = p - 0.5*rho*(phi_c0 - phi_m1);
    const Real q_m = wm2[IP] - (p + 0.5*(rho + wm2[IDN])*(phi_m1 - phi_m2)); //p1 of the left neighbour
    const Real q_p = w0[IP]  - (p - 0.5*(rho + w0[IDN])*(phi_c0 - phi_m1));  //p1 of the right neighbour
    const Vec4 sl = limited_slope(Vec4{q_m, 0.0, 0.0, 0.0}, Vec4{0.0, 0.0, 0.0, 0.0},
                                  Vec4{q_p, 0.0, 0.0, 0.0}, dk, limiter);
    a[IP] = p0_face + 0.5*dk*sl[0];
  }
  //right state (cell c0, extrapolated to its left edge)
  {
    const Real rho = w0[IDN], p = w0[IP];
    const Real p0_face = p + 0.5*rho*(phi_c0 - phi_m1);
    const Real q_m = wm1[IP] - (p + 0.5*(rho + wm1[IDN])*(phi_c0 - phi_m1));
    const Real q_p = wp1[IP] - (p - 0.5*(rho + wp1[IDN])*(phi_p1 - phi_c0));
    const Vec4 sl = limited_slope(Vec4{q_m, 0.0, 0.0, 0.0}, Vec4{0.0, 0.0, 0.0, 0.0},
                                  Vec4{q_p, 0.0, 0.0, 0.0}, dk, limiter);
    b[IP] = p0_face - 0.5*dk*sl[0];
  }
  l = prim_to_cons(a, gamma);
  r = prim_to_cons(b, gamma);
}
