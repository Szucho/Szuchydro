#include "pgen/pgen.hpp"
#include "physics/turbulence.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

//Driven turbulence in a fully periodic box: 
//uniform gas (gas_rho0, gas_p0, optional uniform velocity gas_u0, gas_v0),
//stirred by the few-mode driver in physics/turbulence.*. 
//Two optional split terms:
//  turb_cool_tau > 0: relax the temperature back to its initial value (the driving work would
//                     otherwise heat the box up forever)
//  turb_drag > 0:     linear drag -turb_drag * v ("Ekman" friction). In 2D the energy cascades to
//                     LARGE scales; with forcing at small scales, the drag removes it there, so a
//                     steady inverse cascade can form instead of a box-sized condensate.


void ProblemTurb(Mesh& mesh, const SimParams& par, ProblemHooks& hooks){
  const ParamStore& s = par.store;
  for(const std::string* bc : {&par.bc_left, &par.bc_right, &par.bc_bottom, &par.bc_top})
    if(*bc != "Periodic")
      throw std::runtime_error("Parameter error: gas_profile turb needs Periodic boundaries on all walls");

  const Real gamma = par.gamma;
  const Real rho0 = s.get_real("gas_rho0");
  const Real p0   = s.get_real("gas_p0");
  TurbParams tp;  //defaults from turbulence.hpp, overridden by the parameter file
  tp.k_low      = s.get_int ("turb_k_low", tp.k_low);
  tp.k_peak     = s.get_real("turb_k_peak", tp.k_peak);
  tp.k_high     = s.get_int ("turb_k_high", tp.k_high);
  tp.nmodes     = s.get_int ("turb_nmodes", tp.nmodes);
  tp.tcorr      = s.get_real("turb_tcorr", tp.tcorr);
  tp.accel_rms  = s.get_real("turb_accel_rms", tp.accel_rms);
  tp.sol_weight = s.get_real("turb_sol_weight", tp.sol_weight);
  tp.seed       = (std::uint64_t)s.get_int("turb_seed", (int)tp.seed);
  const Real cool_tau = s.get_real("turb_cool_tau", 0.0);  //0 = no cooling
  const Real drag     = s.get_real("turb_drag", 0.0); //0 = no drag
  const Real u0       = s.get_real("gas_u0", 0.0);
  const Real v0       = s.get_real("gas_v0", 0.0); //init uniform velocity
  if(drag < 0.0) throw std::runtime_error("Parameter error: turb_drag must be >= 0");
  //a forced wavevector must exist on the grid: |n| < N/2 (the highest wavenumber of an N-cell grid)
  if(2*s.get_int("turb_k_high", TurbParams().k_high) >= std::min(par.Nx, par.Ny))
    throw std::runtime_error("Parameter error: turb_k_high must be below half the root grid size "
                             "(Nx/2 = " + std::to_string(std::min(par.Nx, par.Ny)/2) + "),"
                             " otherwise the forcing is aliased");
  

  //initial state: uniform, at rest (unless gas_u0/gas_v0 are given)
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;
  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    const int b_slot = b.slot_id;
    const int total_i = b.block_size.nx1 + 2*nghost, total_j = b.block_size.nx2 + 2*nghost;
    Kokkos::parallel_for("InitTurb", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
      KOKKOS_LAMBDA(int i, int j){
        data(b_slot, i, j, IDN) = rho0;
        data(b_slot, i, j, IM1) = rho0*u0;
        data(b_slot, i, j, IM2) = rho0*v0;
        data(b_slot, i, j, IEN) = p0/(gamma - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);
      });
  }

  //the driver: modes, OU amplitudes, normalization
  auto drv = std::make_shared<TurbulenceDriver>(tp, par.xmin, par.ymin, par.xmax - par.xmin, par.ymax - par.ymin);
  std::cout << "turb: " << drv->modes().size() << " modes:";
  for(const auto& n : drv->modes()) std::cout << " (" << n[0] << "," << n[1] << ")";
  std::cout << "\n";

  hooks.begin_step = [drv](Mesh& m, Real dt){ drv->begin_step(m, dt); };
  hooks.smooth_sources = [drv](Mesh& m, const View4D& Us, const View4D& Uo, Real dt){
    drv->add_forcing(m, Us, Uo, dt);
  };

  //operator-split terms after each step: 
  //drag (exact factor exp(-drag dt) on the momentum,
  //the internal energy is kept, the removed kinetic energy leaves the system),
  //then cooling (internal energy relaxed to rho*T0/(gamma - 1), T0 = p0/rho0)
  if(cool_tau > 0.0 || drag > 0.0){
    const Real T0 = p0/rho0;
    hooks.source_terms = [cool_tau, drag, T0, gamma](Mesh& m, Real dt){
      View4D U = m.get_data_view();
      const int n_active = m.num_active_blocks();
      const int ng = m.inputs().nghost, nx1 = m.inputs().nx1_block, nx2 = m.inputs().nx2_block;
      const Real fd = std::exp(-drag*dt);
      const Real fc = (cool_tau > 0.0) ? 1.0 - std::exp(-dt/cool_tau) : 0.0; //0 if no cooling
      Kokkos::parallel_for("TurbDragCooling",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ng + nx1, ng + nx2}),
        KOKKOS_LAMBDA(const int b, const int i, const int j){
          const Real rho = U(b, i, j, IDN);
          const Real mx = U(b, i, j, IM1), my = U(b, i, j, IM2);
          Real eint = U(b, i, j, IEN) - 0.5*(mx*mx + my*my)/rho;
          const Real mxn = fd*mx, myn = fd*my;
          eint += (rho*T0/(gamma - 1.0) - eint)*fc;
          U(b, i, j, IM1) = mxn;
          U(b, i, j, IM2) = myn;
          U(b, i, j, IEN) = eint + 0.5*(mxn*mxn + myn*myn)/rho;
        });
    };
  }
}
