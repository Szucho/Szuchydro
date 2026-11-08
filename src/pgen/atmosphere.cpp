#include "pgen/pgen.hpp"
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

/*
 Stratified layer in hydrostatic equilibrium in an external potential (gas_profile atmosphere).

 The potential (hooks.potential) varies along one axis s (atm_axis x or y):
   atm_potential uniform    phi = atm_g * s               (constant gravity)
   atm_potential harmonic   phi = 1/2 atm_omega^2 s^2     (vertical gravity near a disk midplane)
 Temperature T = p/rho along s: atm_T0, or a tanh step from atm_T0 to atm_T1 at atm_s_jump with
 width atm_width (e.g. a cold layer under a hot one).
 With atm_symmetric 1 the step uses |s|: a cold midplane with
 a hot layer on BOTH sides (like a disk with a heated surface).

 The initial state is the **DISCRETE** hydrostatic equilibrium the well-balanced scheme preserves:
   p_{j+1} - p_j = -(rho_j + rho_{j+1})/2 (phi_{j+1} - phi_j),   rho = p/T
   => p_{j+1} (1 + dphi/(2 T_{j+1})) = p_j (1 - dphi/(2 T_j))
 built separately for every refinement level (its own cell spacing),
 starting from gas_rho0 at the first cell centre of the domain along s.
 Exact for any T profile :D

 Optional pressure perturbation: p *= 1 + atm_pert exp(-r^2/(2 atm_pert_sigma^2)) around
 (atm_pert_x, atm_pert_y), for testing that small disturbances propagate on top of the equilibrium.
*/
void ProblemAtmosphere(Mesh& mesh, const SimParams& par, ProblemHooks& hooks){
  const ParamStore& s = par.store;
  const Real gamma = par.gamma;
  const Real rho0 = s.get_real("gas_rho0");
  const std::string axis = s.get("atm_axis", "y");
  const std::string pot  = s.get("atm_potential", "uniform");
  const Real g     = s.get_real("atm_g", 1.0);
  const Real omega = s.get_real("atm_omega", 1.0);
  const Real T0    = s.get_real("atm_T0", 1.0);
  const Real T1    = s.get_real("atm_T1", T0);   //= T0: isothermal
  const bool along_x = (axis == "x");
  const Real smin = along_x ? par.xmin : par.ymin, smax = along_x ? par.xmax : par.ymax;
  const Real s_jump = s.get_real("atm_s_jump", 0.5*(smin + smax));
  const Real width  = s.get_real("atm_width", 0.05*(smax - smin));
  const Real pert   = s.get_real("atm_pert", 0.0);
  const Real px     = s.get_real("atm_pert_x", 0.5*(par.xmin + par.xmax));
  const Real py     = s.get_real("atm_pert_y", 0.5*(par.ymin + par.ymax));
  const Real psig   = s.get_real("atm_pert_sigma", 0.05*(smax - smin));
  const bool symmetric = s.get_int("atm_symmetric", 0) != 0;
  if(axis != "x" && axis != "y")
    throw std::runtime_error("Parameter error: atm_axis must be x or y, got '" + axis + "'");
  if(pot != "uniform" && pot != "harmonic")
    throw std::runtime_error("Parameter error: atm_potential must be uniform or harmonic, got '" + pot + "'");
  if(T0 <= 0.0 || T1 <= 0.0) throw std::runtime_error("Parameter error: atm_T0 and atm_T1 must be positive");

  //the potential: depends on the coordinate along the stratification axis only
  auto phi_s = [pot, g, omega](Real sc){ return (pot == "uniform") ? g*sc : 0.5*omega*omega*sc*sc; };
  hooks.potential = [phi_s, along_x](Real x, Real y){ return phi_s(along_x ? x : y); };
  auto T_of = [T0, T1, s_jump, width, symmetric](Real sc){
    const Real d = symmetric ? std::abs(sc) : sc;
    return T0 + (T1 - T0)*0.5*(1.0 + std::tanh((d - s_jump)/width));
  };

  //discrete equilibrium per cell spacing ds (one per refinement level), ghost cells included:
  //p[j + ng] at s_j = smin + (j + 0.5) ds, j = -ng .. N + ng - 1
  const int ng = mesh.inputs().nghost;
  std::map<Real, std::vector<Real>> profiles;
  auto profile = [&](Real ds) -> const std::vector<Real>& {
    auto it = profiles.find(ds);
    if(it != profiles.end()) return it->second;
    const int N = (int)std::lround((smax - smin)/ds);
    std::vector<Real> p(N + 2*ng);
    auto sj = [&](int j){ return smin + (j + 0.5)*ds; };
    p[ng] = rho0*T_of(sj(0));
    for(int j = 0; j < N + ng - 1; ++j){       //upwards
      const Real dphi = phi_s(sj(j + 1)) - phi_s(sj(j));
      p[ng + j + 1] = p[ng + j]*(1.0 - dphi/(2.0*T_of(sj(j))))/(1.0 + dphi/(2.0*T_of(sj(j + 1))));
    }
    for(int j = 0; j > -ng; --j){              //downwards into the lower ghost cells
      const Real dphi = phi_s(sj(j)) - phi_s(sj(j - 1));
      p[ng + j - 1] = p[ng + j]*(1.0 + dphi/(2.0*T_of(sj(j))))/(1.0 - dphi/(2.0*T_of(sj(j - 1))));
    }
    for(Real v : p)
      if(!(v > 0.0)) throw std::runtime_error("Parameter error: atmosphere pressure drops to <= 0 "
                                              "(potential too strong for the temperature, or cells too coarse)");
    return profiles.emplace(ds, std::move(p)).first->second;
  };

  //fill on the host (setup only), then copy to the device
  View4D data = mesh.get_data_view();
  auto h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(h, data);
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    const int slot = b.slot_id;
    const Real dx = b.dx1(), dy = b.dx2();
    const Real ds = along_x ? dx : dy;
    const std::vector<Real>& p = profile(ds);
    for(int i = 0; i < ni; ++i){
      const Real x = b.block_size.x1min + (i - b.is + 0.5)*dx;
      for(int j = 0; j < nj; ++j){
        const Real y = b.block_size.x2min + (j - b.js + 0.5)*dy;
        const Real sc = along_x ? x : y;
        const int jg = (int)std::lround((sc - smin)/ds - 0.5);  //global index along s at this level
        Real pres = p[jg + ng];
        const Real rho = pres/T_of(smin + (jg + 0.5)*ds);
        if(pert != 0.0){
          const Real r2 = (x - px)*(x - px) + (y - py)*(y - py);
          pres *= 1.0 + pert*std::exp(-0.5*r2/(psig*psig));
        }
        h(slot, i, j, IDN) = rho;
        h(slot, i, j, IM1) = 0.0;
        h(slot, i, j, IM2) = 0.0;
        h(slot, i, j, IEN) = pres/(gamma - 1.0);
      }
    }
  }
  Kokkos::deep_copy(data, h);
}
