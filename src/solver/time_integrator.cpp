#include "solver/time_integrator.hpp"
#include <stdexcept>

//throw "Parameter error" if unknown integrator name
TimeIntegrator::TimeIntegrator(const std::string& name, const MeshInputs& in) : name_(name){
  //only Euler for now
  if(name_ != "euler")
    throw std::runtime_error("Parameter error: unknown integrator '" + name_ + "' (euler)");
  F_ = AllocateFaceFluxes(in);
}

//integrator step function
void TimeIntegrator::step(Mesh& mesh, Real dt, Real gamma, const ReconOptions& recon,
                          const SourceFn& smooth_sources){
  if(mesh.num_active_blocks() == 0) return;
  View4D U    = mesh.get_data_view();
  View4D Unew = mesh.get_data_next_view();

  //forward Euler: one stage
  //TODO: VL2 SOON
  ComputeFluxes(mesh, U, gamma, recon, F_);
  ApplyFluxDivergence(mesh, U, Unew, F_, dt);
  if(smooth_sources) smooth_sources(mesh, U, Unew, dt); //easy addition of well behaved source terms

  mesh.swap_data_views(); //swap
}
