#pragma once

#include <functional>
#include <string>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "hydro/reconstruction.hpp"
#include "solver/rhs.hpp"
#include "solver/potential.hpp"

/*
 Time integration: U^n -> U^{n+1} over one step dt.

 Integrators (parameter "integrator"):
   euler   forward Euler: U^{n+1} = U^n + dt L(U^n)                   (default; 1st order in time)
   vl2     van Leer predictor-corrector (Stone & Gardiner 2009), 2nd order in time:
             U*      = U^n + dt/2 L1(U^n)    L1: first-order (piecewise-constant) face states
             U^{n+1} = U^n + dt   L(U*)      L:  the normal second-order reconstruction
   rk2     Heun / SSP-RK2, 2nd order in time:
             U1      = U^n + dt L(U^n)
             U^{n+1} = 1/2 U^n + 1/2 (U1 + dt L(U1))
 Between stages, the integrator calls fill_ghosts (ghost exchange + all boundary conditions).


 L(U) = -div F(U) + S(U)
 S are the **SMOOTH** source terms, evaluated INSIDE the update from the
 same stage state as the fluxes (gravity, forcing, ...):
 with several stages they are part of every stage,
 which is what keeps a well-balanced scheme balanced.
 Stiff, operator-split terms (cooling, relaxation) are not part of L; 
 they run after the step (ProblemHooks::source_terms) operator-split.
*/

//smooth well behaved source terms: add dt * S(U_stage) to the interior cells of U_out
using SourceFn = std::function<void(Mesh& mesh, const View4D& U_stage, const View4D& U_out, Real dt)>;

class TimeIntegrator{
  private:
    std::string name_;
    FaceFluxes F_;
    View3D phi_; //defaults to empty = no external potential field
    bool well_balanced_ = false;
    View4D flux_reg_saved_; //rk2: block-edge fluxes of stage 1 
  
  public:
    //"Parameter error" for unkown name 
    TimeIntegrator(const std::string& name, const MeshInputs& in);
    //advance the active blocks by dt; on return mesh.get_data_view() holds U^{n+1} (interior cells;
    //ghost cells are filled afterwards by the usual ghost exchange and boundary conditions), and the
    //mesh flux register holds the block-edge fluxes of this step for ApplyReflux.
    //fill_ghosts: ghost exchange + all boundary conditions on mesh.get_data_view(); multi-stage
    //integrators call it between their stages (euler never does)
    void step(Mesh& mesh, Real dt, Real gamma, const ReconOptions& recon, const SourceFn& smooth_sources,
              const std::function<void()>& fill_ghosts);

    const std::string& name() const { return name_; }
    
    //external potential (tabulated by PotentialField):
    //its source is part of L(U), evaluated from the stage like the fluxes.
    //well balanced selects the hydrostatic reconstruction
    void set_potential(const View3D& phi, bool well_balanced){ phi_ = phi; well_balanced_ = well_balanced; }
};
