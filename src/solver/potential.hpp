#pragma once

#include <functional>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"

/*
 External potential phi(x, y): any conservative force field entering the Euler (or Navier-Stokes)
 equations as  d(rho v)/dt = ... - rho grad(phi),   dE/dt = ... - rho v . grad(phi).
 Gravity is one example; a centrifugal or tidal potential works the same way.

 The PROBLEM defines phi (ProblemHooks::potential, a host function). GPU kernels cannot call host
 functions, so the solver tabulates phi at every cell centre (ghost cells included) into a device
 array: at setup and after every AMR regrid. All kernels use differences of that array, so a later
 self-gravity solver could fill the same array instead.

 With "well_balanced 1" the fluxes use the hydrostatic reconstruction (reconstruct_face_wb); the
 source term below uses exactly the same phi differences, so discrete equilibria stay at rest to
 machine precision. With "well_balanced 0" (default) the same source is combined with the normal
 reconstruction.
*/

//func specified in ProblemFile.cpp by user
using PotentialFn = std::function<Real(Real x, Real y)>;

class PotentialField{
public:
  //evaluate phi at all cell centres of the active blocks (ghost cells included), allocates on first use
  void fill(Mesh& mesh, const PotentialFn& phi);
  const View3D& view() const { return phi_; }

private:
  View3D phi_;
  View3D::host_mirror_type h_phi_;
};

//add dt * S(U_stage) to the interior of U_out: momentum -rho grad(phi), energy -(rho v).grad(phi),
//centred differences grad(phi)_i = (phi_{i+1} - phi_{i-1})/(2 dx)
void AddPotentialSource(Mesh& mesh, const View3D& phi, const View4D& U_stage, const View4D& U_out, Real dt);

//ghost cells of Hydrostatic walls (after the regular BC has mirrored them like Closed walls):
//density from the adjacent interior cell, pressure extrapolated with phi so that the discrete
//equilibrium holds across the wall, p_ghost = p_in - rho_in (phi_ghost - phi_in)
//the mirrored (reflected) velocity is kept
void ApplyHydrostaticGhosts(Mesh& mesh, const View3D& phi, const GridBC& bc, Real gamma);
