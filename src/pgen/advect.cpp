#include "pgen/pgen.hpp"
#include <cmath>
#include <stdexcept>

//Advected density wave (convergence tests): rho = gas_rho0 (1 + adv_amp sin(2 pi (kx x/Lx + ky y/Ly)))
//with uniform pressure gas_p0 and velocity (gas_u0, gas_v0), in a fully periodic box. The exact
//solution is the initial profile shifted by the velocity times t. Cells get exact CELL AVERAGES of
//the sine (closed form), so a measured error is purely the scheme's.
void ProblemAdvect(Mesh& mesh, const SimParams& par, ProblemHooks& /*hooks*/){
  for(const std::string* bc : {&par.bc_left, &par.bc_right, &par.bc_bottom, &par.bc_top})
    if(*bc != "Periodic")
      throw std::runtime_error("Parameter error: gas_profile advect needs Periodic boundaries on all walls");
  const ParamStore& s = par.store;
  const Real gamma = par.gamma;
  const Real rho0 = s.get_real("gas_rho0"), p0 = s.get_real("gas_p0");
  const Real u0 = s.get_real("gas_u0"), v0 = s.get_real("gas_v0");
  const Real amp = s.get_real("adv_amp", 0.1);
  const int kx = s.get_int("adv_kx", 1), ky = s.get_int("adv_ky", 0);
  const Real ax = 2.0*M_PI*kx/(par.xmax - par.xmin), ay = 2.0*M_PI*ky/(par.ymax - par.ymin);
  const Real x0 = par.xmin, y0 = par.ymin;
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;
  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    const int b_slot = b.slot_id, is = b.is, js = b.js;
    const int total_i = b.block_size.nx1 + 2*nghost, total_j = b.block_size.nx2 + 2*nghost;
    const Real x1min = b.block_size.x1min, x2min = b.block_size.x2min, dx = b.dx1(), dy = b.dx2();
    //cell average of sin(ax x + ay y) = sin(centre) * sinc(ax dx/2) * sinc(ay dy/2)
    const Real hx = 0.5*ax*dx, hy = 0.5*ay*dy;
    const Real fx = (hx != 0.0) ? std::sin(hx)/hx : 1.0, fy = (hy != 0.0) ? std::sin(hy)/hy : 1.0;
    Kokkos::parallel_for("InitAdvect", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
      KOKKOS_LAMBDA(int i, int j){
        const Real x = x1min + (i - is + 0.5)*dx - x0, y = x2min + (j - js + 0.5)*dy - y0;
        const Real rho = rho0*(1.0 + amp*fx*fy*Kokkos::sin(ax*x + ay*y));
        data(b_slot, i, j, IDN) = rho;
        data(b_slot, i, j, IM1) = rho*u0;
        data(b_slot, i, j, IM2) = rho*v0;
        //E = p/(gamma-1) + 1/2 rho |v|^2 is linear in rho, so this is its exact cell average too
        data(b_slot, i, j, IEN) = p0/(gamma - 1.0) + 0.5*rho*(u0*u0 + v0*v0);
      }
    );
  }
}
