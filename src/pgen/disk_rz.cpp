#include "pgen/pgen.hpp"
#include "physics/disk_physics.hpp"
#include <memory>

//Stirred protoplanetary disk with winds in the R-z plane (x = R, y = z); the physics is in
//src/physics/disk_physics.*. Parameters: gas_rho0 (density scale) and the disk keys below.
void ProblemDiskRz(Mesh& mesh, const SimParams& par, ProblemHooks& hooks){
  const ParamStore& s = par.store;
  DiskParams d;  //defaults from disk_physics.hpp, overridden by the parameter file
  d.h_disk      = s.get_real("h_disk", d.h_disk);
  d.h_wind      = s.get_real("h_wind", d.h_wind);
  d.eta_tr      = s.get_real("eta_tr", d.eta_tr);
  d.q_D         = s.get_real("q_D", d.q_D);
  d.rho_floor   = s.get_real("rho_floor", d.rho_floor);
  d.p_floor     = s.get_real("p_floor", d.p_floor);
  d.beta_cool   = s.get_real("beta_cool", d.beta_cool);
  d.stir_nmodes = s.get_int ("stir_nmodes", d.stir_nmodes);
  d.stir_amp    = s.get_real("stir_amp", d.stir_amp);
  d.stir_tcorr  = s.get_real("stir_tcorr", d.stir_tcorr);
  d.stir_damp   = s.get_real("stir_damp", d.stir_damp);
  d.stir_lmin   = s.get_real("stir_lmin", d.stir_lmin);
  d.stir_lmax   = s.get_real("stir_lmax", d.stir_lmax);
  d.stir_seed   = (unsigned)s.get_int("stir_seed", (int)d.stir_seed);
  d.wind_mach   = s.get_real("wind_mach", d.wind_mach);
  d.wind_angle  = s.get_real("wind_angle", d.wind_angle);
  d.wind_tau    = s.get_real("wind_tau", d.wind_tau);
  d.corona_tau  = s.get_real("corona_tau", d.corona_tau);
  const Real rho0 = s.get_real("gas_rho0");
  const Real gamma = par.gamma;

  //disk model with an |eta| table covering the whole domain incl. ghost cells
  Real ymax_abs = std::max(std::abs(par.ymin), std::abs(par.ymax));
  Real eta_max = 1.5*ymax_abs/par.xmin + 1.0;
  auto model = std::make_shared<DiskModel>(BuildDiskModel(d, rho0, eta_max));
  InitDiskRz(mesh, *model, gamma);

  //time loop extras: stirring, wind, cooling (split), and the paper-style ghost-cell reset.
  //shared_ptr: the hooks own the model and the stirring state for the whole run
  auto stir = std::make_shared<StirringForce>(d);
  hooks.source_terms = [model, stir, gamma](Mesh& m, Real dt){
    stir->Advance(dt);
    AddDiskSourceTerms(m, *model, *stir, dt, gamma);
  };
  hooks.boundaries = [model, gamma](Mesh& m){
    ApplyDiskBoundaries(m, *model, gamma);
  };
}
