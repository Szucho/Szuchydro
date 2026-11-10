#include "solver/time_integrator.hpp"
#include "solver/potential.hpp"
#include <stdexcept>

//throw "Parameter error" if unknown integrator name
TimeIntegrator::TimeIntegrator(const std::string& name, const MeshInputs& in) : name_(name){
  if(name_ != "euler" && name_ != "vl2" && name_ != "rk2")
    throw std::runtime_error("Parameter error: unknown integrator '" + name_ + "' (euler, vl2, rk2)");
  F_ = AllocateFaceFluxes(in);
}

//integrator step function
void TimeIntegrator::step(Mesh& mesh, Real dt, Real gamma, const ReconOptions& recon,
                          const SourceFn& smooth_sources, const std::function<void()>& fill_ghosts){
  if(mesh.num_active_blocks() == 0) return;
  const bool has_phi = phi_.extent(0) > 0;

  //the source terms of L(U) evaluated from U_stage, added to U_out with time step h
  auto add_sources = [&](const View4D& U_stage, const View4D& U_out, Real h){
    if(has_phi) AddPotentialSource(mesh, phi_, U_stage, U_out, h); //the external potential
    if(smooth_sources) smooth_sources(mesh, U_stage, U_out, h);    //easy addition of well behaved source terms
  };

  if(name_ == "euler"){
    //forward Euler: one stage
    View4D U    = mesh.get_data_view();
    View4D Unew = mesh.get_data_next_view();
    ComputeFluxes(mesh, U, gamma, recon, F_, phi_, well_balanced_);
    ApplyFluxDivergence(mesh, U, Unew, F_, dt);
    add_sources(U, Unew, dt);
    mesh.swap_data_views();
    return;
  }

  if(name_ == "vl2"){
    //predictor: U* = U^n + dt/2 L1(U^n), first-order face states, into the second array
    {
      View4D Un = mesh.get_data_view(), Us = mesh.get_data_next_view();
      ComputeFluxes(mesh, Un, gamma, recon, F_, phi_, well_balanced_, /*first_order=*/true);
      ApplyFluxDivergence(mesh, Un, Us, F_, 0.5*dt);
      add_sources(Un, Us, 0.5*dt);
    }
    mesh.swap_data_views(); //data = U*, data_next = U^n
    fill_ghosts();          //ghost exchange + boundaries on U* (the corrector's stencils need them)
    //corrector: U^{n+1} = U^n + dt L(U*), written over U^n (every cell only reads its own value)
    {
      View4D Us = mesh.get_data_view(), Un = mesh.get_data_next_view();
      ComputeFluxes(mesh, Us, gamma, recon, F_, phi_, well_balanced_); //also fills the flux register
      ApplyFluxDivergence(mesh, Un, Un, F_, dt);
      add_sources(Us, Un, dt);
    }
    mesh.swap_data_views(); //data = U^{n+1}
    return;
  }

  //rk2 (Heun / SSP-RK2)
  View4D reg = mesh.get_flux_view();
  if(flux_reg_saved_.extent(0) != reg.extent(0)){
    flux_reg_saved_ = View4D("flux_reg_saved", reg.extent(0), reg.extent(1), reg.extent(2), reg.extent(3));
  }
  //stage 1: U1 = U^n + dt L(U^n)
  {
    View4D Un = mesh.get_data_view(), U1 = mesh.get_data_next_view();
    ComputeFluxes(mesh, Un, gamma, recon, F_, phi_, well_balanced_);
    ApplyFluxDivergence(mesh, Un, U1, F_, dt);
    add_sources(Un, U1, dt);
  }
  Kokkos::deep_copy(flux_reg_saved_, reg); //stage-1 block-edge fluxes
  mesh.swap_data_views(); //data = U1, data_next = U^n
  fill_ghosts();
  //stage 2: U^{n+1} = 1/2 U^n + 1/2 (U1 + dt L(U1)), written over U^n
  {
    View4D U1 = mesh.get_data_view(), Un = mesh.get_data_next_view();
    ComputeFluxes(mesh, U1, gamma, recon, F_, phi_, well_balanced_);
    CombineFluxDivergence(mesh, Un, U1, Un, F_, dt, 0.5, 0.5);
    add_sources(U1, Un, 0.5*dt);
  }
  //U^{n+1} = U^n + dt/2 (L(U^n) + L(U1)): the effective flux is the stage average, and the
  //refluxing (ApplyReflux with dt) must see exactly that
  {
    const View4D saved = flux_reg_saved_;
    Kokkos::parallel_for("Rk2AverageFluxRegister",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
        {(int)reg.extent(0), (int)reg.extent(1), (int)reg.extent(2), (int)reg.extent(3)}),
      KOKKOS_LAMBDA(const int b, const int f, const int t, const int k){
        reg(b, f, t, k) = 0.5*(saved(b, f, t, k) + reg(b, f, t, k));
      });
  }
  mesh.swap_data_views(); //data = U^{n+1}
}
