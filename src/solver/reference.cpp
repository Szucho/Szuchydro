#include "solver/reference.hpp"

void ReferenceEquilibrium::capture(Mesh& mesh, Real gamma){
  const int ng = mesh.inputs().nghost;
  const int nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int ni = nx1 + 2*ng, nj = nx2 + 2*ng;
  if(rho_.extent(0) == 0){
    rho_ = View3D("reference_rho", mesh.inputs().max_blocks, ni, nj);
    p_   = View3D("reference_p",   mesh.inputs().max_blocks, ni, nj);
  }
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  View4D U = mesh.get_data_view();
  const View3D rho = rho_, p = p_;
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj});

  //the problem's own values, every cell: at level boundaries these ghost cells are exactly the
  //discrete equilibrium of the block's own level, which is what the deviation exchange needs
  Kokkos::parallel_for("ReferenceCapture", policy,
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const Real r = U(b, i, j, IDN), mx = U(b, i, j, IM1), my = U(b, i, j, IM2);
      rho(b, i, j) = r;
      p(b, i, j) = (gamma - 1.0)*(U(b, i, j, IEN) - 0.5*(mx*mx + my*my)/r);
    });

  //wherever the exchange fills a ghost cell by a same-level COPY (neighbouring blocks, and the
  //wrap-around of periodic boundaries), the reference ghost must equal the neighbour's reference
  //EXACTLY: the exchange makes ghost = (U_nb - ref_nb) + ref_ghost, which is U_nb only then.
  //The problem setup cannot guarantee that (it evaluates a periodic ghost at its position outside
  //the domain, not at its periodic image), and a mismatch makes the two blocks compute a shared
  //face flux from different data, so mass leaks. One ordinary exchange of the initial state
  //provides exact copies; they replace the reference on same-level faces only.
  ExchangeGhostZones(mesh);
  IntView2D n_level = mesh.get_neighbor_level_view();
  const int is = ng, ie = ng + nx1 - 1, js = ng, je = ng + nx2 - 1;
  Kokkos::parallel_for("ReferenceSameLevelGhosts", policy,
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      int face = -1; //which face's ghost region (i, j) belongs to; corners and interior: none
      if(j >= js && j <= je){ if(i < is) face = (int)FaceDir::XM; else if(i > ie) face = (int)FaceDir::XP; }
      if(i >= is && i <= ie){ if(j < js) face = (int)FaceDir::YM; else if(j > je) face = (int)FaceDir::YP; }
      if(face < 0 || n_level(b, face) != (int)NeighborLevel::Same) return;
      const Real r = U(b, i, j, IDN), mx = U(b, i, j, IM1), my = U(b, i, j, IM2);
      rho(b, i, j) = r;
      p(b, i, j) = (gamma - 1.0)*(U(b, i, j, IEN) - 0.5*(mx*mx + my*my)/r);
    });
}

void ReferenceEquilibrium::to_deviation(Mesh& mesh, Real gamma) const{
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  View4D U = mesh.get_data_view();
  const View3D rho = rho_, p = p_;
  Kokkos::parallel_for("ToDeviation",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      U(b, i, j, IDN) -= rho(b, i, j);
      U(b, i, j, IEN) -= p(b, i, j)/(gamma - 1.0);  //the reference is at rest: no momentum, no kinetic energy
    });
}

void ReferenceEquilibrium::from_deviation(Mesh& mesh, Real gamma) const{
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  View4D U = mesh.get_data_view();
  const View3D rho = rho_, p = p_;
  Kokkos::parallel_for("FromDeviation",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      U(b, i, j, IDN) += rho(b, i, j);
      U(b, i, j, IEN) += p(b, i, j)/(gamma - 1.0);
    });
}
