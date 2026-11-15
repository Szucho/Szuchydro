#include "pgen/pgen.hpp"
#include <cmath>
#include <stdexcept>

/*
 Atmosphere around a softened point mass (gas_profile pointmass): a genuinely 2D potential
   phi = -pm_GM / sqrt((x - pm_x)^2 + (y - pm_y)^2 + pm_eps^2)
 with a barotropic equilibrium (p and rho functions of phi alone), in closed form:
   pm_equilibrium isothermal:  rho = gas_rho0 exp(-(phi - phi_c)/pm_T),  p = rho pm_T
   pm_equilibrium polytropic:  p = K rho^G (G = pm_gamma, default gamma), enthalpy
                               h = G/(G-1) p/rho = G/(G-1) pm_T - (phi - phi_c)
 gas_rho0 and pm_T are the density and temperature at the mass (phi_c = phi at the centre)
 Used with well_balanced 1 and the matching wb_equilibrium, the state stays at rest to machine precision
 Optional pressure pulse: p *= 1 + pm_pert exp(-r^2/(2 pm_pert_sigma^2)) around (pm_pert_x, pm_pert_y).
*/
void ProblemPointmass(Mesh& mesh, const SimParams& par, ProblemHooks& hooks){
  const ParamStore& s = par.store;
  const Real gamma = par.gamma;
  const Real rho0  = s.get_real("gas_rho0");
  const Real GM    = s.get_real("pm_GM", 1.0);
  const Real eps   = s.get_real("pm_eps", 0.3);
  const Real xc    = s.get_real("pm_x", 0.5*(par.xmin + par.xmax));
  const Real yc    = s.get_real("pm_y", 0.5*(par.ymin + par.ymax));
  const Real Tc    = s.get_real("pm_T", 1.0);
  const std::string eqn = s.get("pm_equilibrium", "isothermal");
  const bool poly = (eqn == "polytropic");
  const Real G     = poly ? s.get_real("pm_gamma", gamma) : 0.0;
  const Real pert  = s.get_real("pm_pert", 0.0);
  const Real px    = s.get_real("pm_pert_x", xc);
  const Real py    = s.get_real("pm_pert_y", yc);
  const Real psig  = s.get_real("pm_pert_sigma", 0.05*(par.xmax - par.xmin));
  if(eqn != "isothermal" && eqn != "polytropic")
    throw std::runtime_error("Parameter error: pm_equilibrium must be isothermal or polytropic, got '" + eqn + "'");
  if(eps <= 0.0) throw std::runtime_error("Parameter error: pm_eps must be positive (softening)");
  if(poly && G <= 1.0) throw std::runtime_error("Parameter error: pm_gamma must be > 1");

  auto phi = [GM, eps, xc, yc](Real x, Real y){
    return -GM/std::sqrt((x - xc)*(x - xc) + (y - yc)*(y - yc) + eps*eps);
  };
  hooks.potential = phi;
  const Real phic = phi(xc, yc);

  //rho, p of the equilibrium at (x, y)
  auto state = [&](Real x, Real y, Real& rho, Real& p){
    const Real dphi = phi(x, y) - phic;
    if(!poly){
      rho = rho0*std::exp(-dphi/Tc);
      p = rho*Tc;
    }else{
      const Real hc = G/(G - 1.0)*Tc, h = hc - dphi;
      if(!(h > 0.0)) throw std::runtime_error("Parameter error: the polytropic atmosphere ends inside the "
                                              "domain (enthalpy <= 0); raise pm_T or shrink the domain");
      rho = rho0*std::pow(h/hc, 1.0/(G - 1.0));
      p = rho0*Tc*std::pow(h/hc, G/(G - 1.0));
    }
  };

  View4D data = mesh.get_data_view();
  auto hd = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(hd, data);
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    for(int i = 0; i < ni; ++i){
      const Real x = b.block_size.x1min + (i - b.is + 0.5)*b.dx1();
      for(int j = 0; j < nj; ++j){
        const Real y = b.block_size.x2min + (j - b.js + 0.5)*b.dx2();
        Real rho, p;
        state(x, y, rho, p);
        if(pert != 0.0) p *= 1.0 + pert*std::exp(-0.5*((x - px)*(x - px) + (y - py)*(y - py))/(psig*psig));
        hd(b.slot_id, i, j, IDN) = rho;
        hd(b.slot_id, i, j, IM1) = 0.0;
        hd(b.slot_id, i, j, IM2) = 0.0;
        hd(b.slot_id, i, j, IEN) = p/(gamma - 1.0);
      }
    }
  }
  Kokkos::deep_copy(data, hd);
}
