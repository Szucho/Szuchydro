#include "pgen/pgen.hpp"

//uniform state everywhere (ghost cells included)
void ProblemUniform(Mesh& mesh, const SimParams& par, ProblemHooks& /*hooks*/){
  const Real gamma = par.gamma;
  const Real rho0 = par.store.get_real("gas_rho0");
  const Real p0   = par.store.get_real("gas_p0");
  const Real u0   = par.store.get_real("gas_u0");
  const Real v0   = par.store.get_real("gas_v0");
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;

  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int b_slot = b.slot_id;
    int total_i = b.block_size.nx1 + 2*nghost;
    int total_j = b.block_size.nx2 + 2*nghost;
    Kokkos::parallel_for("InitUniform", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
      KOKKOS_LAMBDA(int i, int j) {
        data(b_slot, i, j, IDN) = rho0;
        data(b_slot, i, j, IM1) = rho0*u0;
        data(b_slot, i, j, IM2) = rho0*v0;
        data(b_slot, i, j, IEN) = p0/(gamma - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);
      }
    );
  }
}
