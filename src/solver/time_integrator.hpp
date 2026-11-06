#pragma once

#include <functional>
#include <string>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "hydro/reconstruction.hpp"
#include "solver/rhs.hpp"

/*
 Time integration: U^n -> U^{n+1} over one step dt.

 Integrators (parameter "integrator"):
   euler   forward Euler: U^{n+1} = U^n + dt L(U^n)                   (default; 1st order in time)
   (vl2    van Leer predictor-corrector, 2nd order in time: SOON, TODO)

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
  
  public:
    //"Parameter error" for unkown name 
    TimeIntegrator(const std::string& name, const MeshInputs& in);
    //advance the active blocks by dt; on return mesh.get_data_view() holds U^{n+1} (interior cells;
    //ghost cells are filled afterwards by the usual ghost exchange and boundary conditions), and the
    //mesh flux register holds the block-edge fluxes of this step for ApplyReflux
    void step(Mesh& mesh, Real dt, Real gamma, const ReconOptions& recon, const SourceFn& smooth_sources);

    const std::string& name() const { return name_; }
};
