#include "solver/potential.hpp"

void PotentialField::fill(Mesh& mesh, const PotentialFn& phi){ //fill potential array
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  if(phi_.extent(0) == 0){
    phi_ = View3D("potential", mesh.inputs().max_blocks, ni, nj);
    h_phi_ = Kokkos::create_mirror_view(phi_);
  }
  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    const int s = b.slot_id;
    const Real dx = b.dx1(), dy = b.dx2();
    for(int i = 0; i < ni; ++i){
      const Real x = b.block_size.x1min + (i - b.is + 0.5)*dx; //cell centers
      for(int j = 0; j < nj; ++j){
        const Real y = b.block_size.x2min + (j - b.js + 0.5)*dy;
        h_phi_(s, i, j) = phi(x, y); //evaluate phi at cell centers and fill array
      }
    }
  }
  Kokkos::deep_copy(phi_, h_phi_);
}

//the potential has to work as a source term
void AddPotentialSource(Mesh& mesh, const View3D& phi, const View4D& U_stage, const View4D& U_out, Real dt){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  Kokkos::parallel_for("PotentialSource",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ng + nx1, ng + nx2}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const Real ax = (phi(b, i + 1, j) - phi(b, i - 1, j))/(2.0*block_dx1(b)); //acceleration componets
      const Real ay = (phi(b, i, j + 1) - phi(b, i, j - 1))/(2.0*block_dx2(b)); //coming from grad Phi
      const Real rho = U_stage(b, i, j, IDN);
      U_out(b, i, j, IM1) -= dt*rho*ax; //rho ax dt for time integration
      U_out(b, i, j, IM2) -= dt*rho*ay;
      U_out(b, i, j, IEN) -= dt*(U_stage(b, i, j, IM1)*ax + U_stage(b, i, j, IM2)*ay); //energy
    }
  );
}

//ghost cell (gi, gj) of a Hydrostatic wall from the adjacent interior cell (ii, jj)
KOKKOS_INLINE_FUNCTION
void hydrostatic_ghost(const View4D& U, const View3D& phi, Real gamma,
                       int b, int gi, int gj, int ii, int jj){
  const Real rho_in = U(b, ii, jj, IDN);
  const Real p_in = (gamma - 1.0)*(U(b, ii, jj, IEN) - 0.5*(U(b, ii, jj, IM1)*U(b, ii, jj, IM1)
                                                          + U(b, ii, jj, IM2)*U(b, ii, jj, IM2))/rho_in);
  const Real rho_old = U(b, gi, gj, IDN);
  const Real u = U(b, gi, gj, IM1)/rho_old, v = U(b, gi, gj, IM2)/rho_old; //mirrored, reflected
  const Real p = p_in - rho_in*(phi(b, gi, gj) - phi(b, ii, jj));
  U(b, gi, gj, IDN) = rho_in;
  U(b, gi, gj, IM1) = rho_in*u;
  U(b, gi, gj, IM2) = rho_in*v;
  U(b, gi, gj, IEN) = p/(gamma - 1.0) + 0.5*rho_in*(u*u + v*v);
}

void ApplyHydrostaticGhosts(Mesh& mesh, const View3D& phi, const GridBC& bc, Real gamma){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const bool wall[4] = {bc.left.type == BCType::Hydrostatic, bc.right.type == BCType::Hydrostatic,
                        bc.bottom.type == BCType::Hydrostatic, bc.top.type == BCType::Hydrostatic};
  if(!(wall[0] || wall[1] || wall[2] || wall[3])) return;

  //per active block: which faces are Hydrostatic physical walls (XM, XP, YM, YP)
  Kokkos::View<int*[4], MemorySpace> hw("hydrostatic_walls", n_active);
  auto h_hw = Kokkos::create_mirror_view(hw);  //mirror: same layout as the device view
  const auto& blocks = mesh.blocks();
  for(int b = 0; b < n_active; ++b)
    for(int f = 0; f < 4; ++f) h_hw(b, f) = (blocks[b]->is_physical_boundary[f] && wall[f]) ? 1 : 0;
  Kokkos::deep_copy(hw, h_hw);

  View4D U = mesh.get_data_view();
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int is = ng, ie = ng + nx1 - 1, js = ng, je = ng + nx2 - 1;


  //x-walls: ghost columns, interior rows; y-walls: ghost rows, interior columns
  Kokkos::parallel_for("HydrostaticGhostsX",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, js, 0}, {n_active, je + 1, ng}),
    KOKKOS_LAMBDA(const int b, const int j, const int g){
      if(hw(b, 0)) hydrostatic_ghost(U, phi, gamma, b, is - 1 - g, j, is, j);
      if(hw(b, 1)) hydrostatic_ghost(U, phi, gamma, b, ie + 1 + g, j, ie, j);
    });
  Kokkos::parallel_for("HydrostaticGhostsY",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, 0}, {n_active, ie + 1, ng}),
    KOKKOS_LAMBDA(const int b, const int i, const int g){
      if(hw(b, 2)) hydrostatic_ghost(U, phi, gamma, b, i, js - 1 - g, i, js);
      if(hw(b, 3)) hydrostatic_ghost(U, phi, gamma, b, i, je + 1 + g, i, je);
    });
}
