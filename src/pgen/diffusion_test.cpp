#include "pgen/pgen.hpp"
#include <cmath>
#include <stdexcept>

/*
 Single-mode states for the diffusion tests (gas_profile diffusion_test).
 Uniform background gas_rho0, gas_p0 at rest
 one Fourier mode of amplitude diff_amp and mode number diff_n
 (k = 2 pi diff_n / L along y; along both axes for the diagonal modes):
   shear        u = A sin(k y)                         decays as exp(-nu k^2 t)
   shear_cos    u = A cos(k y)                         the mode that fits free-slip walls in y
   shear_diag   u = A sin(kx x + ky y), v = -(kx/ky) u  divergence-free, decays as exp(-nu |k|^2 t)
   entropy      T = T0 (1 + A sin(k y)) at constant p  decays as exp(-chi k^2 t) (isobaric)
   entropy_cos  T = T0 (1 + A cos(k y))                the mode that fits insulating walls in y
   entropy_diag T = T0 (1 + A sin(kx x + ky y))
   sound        an acoustic wave travelling in +y      decays as exp(-k^2 (4/3 nu + nu_b + (gamma - 1)chi) t/2)
   sound_standing  the same perturbation at rest       a standing wave: kinetic and internal energy slosh
 A is relative to the sound speed for velocities, and relative for T.
*/


void ProblemDiffusionTest(Mesh& mesh, const SimParams& par, ProblemHooks&){
  const ParamStore& s = par.store;
  const Real gamma = par.gamma;
  const Real rho0 = s.get_real("gas_rho0"), p0 = s.get_real("gas_p0");
  const Real A = s.get_real("diff_amp", 1e-4);
  const int n = (int)s.get_real("diff_n", 1.0);
  const std::string mode = s.get("diff_mode", "shear");
  const bool ok = mode == "shear" || mode == "shear_cos" || mode == "shear_diag" || mode == "entropy" ||
                  mode == "entropy_cos" || mode == "entropy_diag" || mode == "sound" || mode == "sound_standing";
  if(!ok) throw std::runtime_error("Parameter error: unknown diff_mode '" + mode + "'");
  const Real T0 = p0/rho0, cs = std::sqrt(gamma*T0);
  const Real kx = 2.0*M_PI*n/(par.xmax - par.xmin), ky = 2.0*M_PI*n/(par.ymax - par.ymin);

  View4D data = mesh.get_data_view();
  auto h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(h, data);
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  for(const auto& bp : mesh.blocks()){
    const MeshBlock& b = *bp;
    for(int i = 0; i < ni; ++i){
      const Real x = b.block_size.x1min + (i - b.is + 0.5)*b.dx1();
      for(int j = 0; j < nj; ++j){
        const Real y = b.block_size.x2min + (j - b.js + 0.5)*b.dx2();
        Real rho = rho0, u = 0.0, v = 0.0, p = p0;
        //the mode's shape: phase along y, or along the diagonal
        const Real ph = (mode == "shear_diag" || mode == "entropy_diag") ? kx*(x - par.xmin) + ky*(y - par.ymin)
                                                                          : ky*(y - par.ymin);
        //the test cases
        if(mode == "shear") u = A*cs*std::sin(ph);
        if(mode == "shear_cos") u = A*cs*std::cos(ph);
        if(mode == "shear_diag"){ u = A*cs*std::sin(ph); v = -(kx/ky)*u; }
        if(mode == "entropy" || mode == "entropy_diag") rho = p0/(T0*(1.0 + A*std::sin(ph)));
        if(mode == "entropy_cos") rho = p0/(T0*(1.0 + A*std::cos(ph)));
        if(mode == "sound_standing"){ //density/pressure mode at rest: two counter-propagating waves
          const Real d = A*std::sin(ph);
          rho = rho0*(1.0 + d);
          p = p0*(1.0 + gamma*d);
        }
        if(mode == "sound"){
          const Real d = A*std::sin(ph);
          rho = rho0*(1.0 + d);
          p = p0*(1.0 + gamma*d);
          v = cs*d;
        }
        //applying them
        h(b.slot_id, i, j, IDN) = rho;
        h(b.slot_id, i, j, IM1) = rho*u;
        h(b.slot_id, i, j, IM2) = rho*v;
        h(b.slot_id, i, j, IEN) = p/(gamma - 1.0) + 0.5*rho*(u*u + v*v);
      }
    }
  }
  Kokkos::deep_copy(data, h);
}
