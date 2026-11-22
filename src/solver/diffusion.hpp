#pragma once

#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "solver/rhs.hpp"

/*
 Explicit diffusive fluxes: viscosity and thermal conduction.

 Coefficients (all constant, default 0 = off):
   nu       kinematic shear viscosity, mu = rho nu at each face
   nu_bulk  kinematic bulk viscosity, mu_b = rho nu_bulk
   chi      thermal diffusivity, heat flux q = -rho c_p chi grad T, T = p/rho, c_p = gamma/(gamma - 1)
 Stress (2D): tau = mu (grad u + grad u^T - 2/3 div(u) I) + mu_b div(u) I.
 Face fluxes added to the hyperbolic ones: momentum -tau.n, energy -(tau.u).n + q.n.

 Face gradients: the normal derivative from the two cells sharing the face; the tangential one from
 the average of the two cells' central differences, one-sided in a block's first and last row
 (column), so no corner ghost cells are needed and neighbouring blocks compute identical face fluxes.

 The fluxes go into the face arrays AND the flux register right after the hyperbolic fluxes, so
 conservation and refluxing work as for the hyperbolic part, with every integrator.
 Reflecting walls (Closed, Hydrostatic) are free-slip and insulating: no diffusive flux crosses them.

 Time step: an explicit diffusion step is stable for D dt (1/dx^2 + 1/dy^2) <= 1/2, with
 D = max(4/3 nu + nu_bulk, gamma chi) (compressive modes; temperature diffuses with gamma chi at
 constant volume); the run uses cfl_diff times this limit.
 
 TODO: SUPERTIMESTEPPING
*/
struct DiffusionCoeffs{
  //holds nu, nubulk, chi
  //active says wheter any isn't 0
  //if inactive diffusion is skipped
  Real nu = 0.0;
  Real nu_bulk = 0.0;
  Real chi = 0.0;
  bool active() const { return nu > 0.0 || nu_bulk > 0.0 || chi > 0.0; }
};

//faces of the active blocks that are reflecting physical walls (Closed or Hydrostatic); empty if none
WallFlags ReflectingWallFlags(Mesh& mesh, const GridBC& bc);

//adds the viscous and conductive fluxes of the state U to F and to the flux register
void AddDiffusiveFluxes(Mesh& mesh, const View4D& U, Real gamma, const DiffusionCoeffs& d,
                        const FaceFluxes& F, const WallFlags& walls);

//cfl_diff times the explicit stability limit, minimum over the active blocks
Real DiffusiveTimeStep(Mesh& mesh, Real gamma, const DiffusionCoeffs& d, Real cfl_diff);
