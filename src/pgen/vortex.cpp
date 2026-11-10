#include "pgen/pgen.hpp"
#include <cmath>
#include <stdexcept>

//Isentropic vortex (Yee et al. 1999; convergence tests): background rho = p = 1, velocity
//(gas_u0, gas_v0), plus the exact steady vortex of strength vortex_beta centred at
//(vortex_x, vortex_y):
//  T = p/rho = 1 - (gamma-1) beta^2/(8 gamma pi^2) exp(1 - r^2),  rho = T^(1/(gamma-1)),  p = rho^gamma
//  du = -beta/(2 pi) exp((1 - r^2)/2) (y - yc),   dv = beta/(2 pi) exp((1 - r^2)/2) (x - xc)
//It is carried unchanged by the background flow, so the exact solution is the shifted initial state.
//Cells get 3x3 Gauss cell averages of the conserved variables. Fully periodic box; it should be
//large enough for exp(-r^2) to vanish at the edges (the classic domain is [-5, 5]^2).
void ProblemVortex(Mesh& mesh, const SimParams& par, ProblemHooks& /*hooks*/){
  for(const std::string* bc : {&par.bc_left, &par.bc_right, &par.bc_bottom, &par.bc_top})
    if(*bc != "Periodic")
      throw std::runtime_error("Parameter error: gas_profile vortex needs Periodic boundaries on all walls");
  const ParamStore& s = par.store;
  const Real gamma = par.gamma;
  const Real u0 = s.get_real("gas_u0"), v0 = s.get_real("gas_v0");
  const Real beta = s.get_real("vortex_beta", 5.0);
  const Real xc = s.get_real("vortex_x", 0.5*(par.xmin + par.xmax));
  const Real yc = s.get_real("vortex_y", 0.5*(par.ymin + par.ymax));
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;
  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    const int b_slot = b.slot_id, is = b.is, js = b.js;
    const int total_i = b.block_size.nx1 + 2*nghost, total_j = b.block_size.nx2 + 2*nghost;
    const Real x1min = b.block_size.x1min, x2min = b.block_size.x2min, dx = b.dx1(), dy = b.dx2();
    Kokkos::parallel_for("InitVortex", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
      KOKKOS_LAMBDA(int i, int j){
        const Real gp[3] = {-0.7745966692414834, 0.0, 0.7745966692414834}; //Gauss points on [-1, 1]
        const Real gw[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};
        Real q[4] = {0.0, 0.0, 0.0, 0.0};
        for(int a = 0; a < 3; ++a){
          for(int c = 0; c < 3; ++c){
            const Real x = x1min + (i - is + 0.5 + 0.5*gp[a])*dx - xc;
            const Real y = x2min + (j - js + 0.5 + 0.5*gp[c])*dy - yc;
            const Real r2 = x*x + y*y;
            const Real T = 1.0 - (gamma - 1.0)*beta*beta/(8.0*gamma*M_PI*M_PI)*Kokkos::exp(1.0 - r2);
            const Real rho = Kokkos::pow(T, 1.0/(gamma - 1.0));
            const Real f = beta/(2.0*M_PI)*Kokkos::exp(0.5*(1.0 - r2));
            const Real u = u0 - f*y, v = v0 + f*x;
            const Real p = Kokkos::pow(rho, gamma);
            const Real w = 0.25*gw[a]*gw[c];
            q[0] += w*rho; q[1] += w*rho*u; q[2] += w*rho*v;
            q[3] += w*(p/(gamma - 1.0) + 0.5*rho*(u*u + v*v));
          }
        }
        data(b_slot, i, j, IDN) = q[0];
        data(b_slot, i, j, IM1) = q[1];
        data(b_slot, i, j, IM2) = q[2];
        data(b_slot, i, j, IEN) = q[3];
      }
    );
  }
}
