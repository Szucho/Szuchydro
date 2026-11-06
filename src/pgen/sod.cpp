#include "pgen/pgen.hpp"

//Sod shock tube along x: left (gas_rho0, gas_p0), right (gas_rho0/8, gas_p0/10), both at rest
void ProblemSodX(Mesh& mesh, const SimParams& par, ProblemHooks& /*hooks*/){
  const Real gamma = par.gamma;
  const Real rho0 = par.store.get_real("gas_rho0");
  const Real p0   = par.store.get_real("gas_p0");
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;

  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int b_slot = b.slot_id;
    int total_i = b.block_size.nx1 + 2*nghost;
    int total_j = b.block_size.nx2 + 2*nghost;
    Real rho_L = rho0,     p_L = p0,     u_L = 0.0, v_L = 0.0;
    Real rho_R = rho0/8.0, p_R = 0.1*p0, u_R = 0.0, v_R = 0.0;
    Real x_mid = 0.5 * (par.xmin + par.xmax);
    int is = b.is; //offset to map cell index to physical coordinates
    Real x1min = b.block_size.x1min;
    Real dx1 = b.dx1();

    //initialize across the ENTIRE block including ghosts so no cell is left with NaN/zeros
    Kokkos::parallel_for("InitSod", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
      KOKKOS_LAMBDA(int i, int j) {
        Real x = x1min + (i - is + 0.5)*dx1;
        Real rho = (x < x_mid) ? rho_L : rho_R;
        Real p   = (x < x_mid) ? p_L   : p_R;
        Real u   = (x < x_mid) ? u_L   : u_R;
        Real v   = (x < x_mid) ? v_L   : v_R;
        Real E   = p/(gamma - 1.0) + 0.5*rho*(u*u + v*v);

        data(b_slot, i, j, IDN) = rho;
        data(b_slot, i, j, IM1) = rho*u;
        data(b_slot, i, j, IM2) = rho*v;
        data(b_slot, i, j, IEN) = E;
      }
    );
  }
}
