#include <stdexcept>
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

//helper namespace
namespace {
  KOKKOS_INLINE_FUNCTION
  int ghost_face(int i, int j, int is, int ie, int js, int je, int& si, int& sj){
    si = i;
    sj = j;
    if(j >= js && j <= je){
      if(i < is){ si = is; return (int)FaceDir::XM; }
      if(i > ie){ si = ie; return (int)FaceDir::XP; }
    }
    if(i >= is && i <= ie){
      if(j < js){ sj = js; return (int)FaceDir::YM; }
      if(j > je){ sj = je; return (int)FaceDir::YP; }
    }
    return -1;
  }
}

void ReferenceEquilibrium::sync_same_level_ghosts(Mesh& mesh, Real gamma){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  //ghosts 2x since they are on both sides
  const int ni = nx1 + 2*ng, nj = nx2 + 2*ng;
  const int is = ng, ie = ng + nx1 - 1, js = ng, je = ng + nx2 - 1;
  const View3D rho = rho_, p = p_; //reference pressure and density
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj});
  View4D S = mesh.get_data_next_view();
  Kokkos::parallel_for("ReferenceToScratch", policy, KOKKOS_LAMBDA(const int b, const int i, const int j){
    S(b, i, j, IDN) = rho(b, i, j);
    S(b, i, j, IM1) = 0.0;
    S(b, i, j, IM2) = 0.0; //no momentum in equilibrium
    S(b, i, j, IEN) = p(b, i, j)/(gamma - 1.0);
  });
  mesh.swap_data_views();
  ExchangeGhostZones(mesh);
  mesh.swap_data_views();
  IntView2D n_level = mesh.get_neighbor_level_view();
  Kokkos::parallel_for("ReferenceSyncSameLevel", policy, KOKKOS_LAMBDA(const int b, const int i, const int j){
    int si, sj;
    const int face = ghost_face(i, j, is, ie, js, je, si, sj);
    if(face < 0 || n_level(b, face) != (int)NeighborLevel::Same) return;
    rho(b, i, j) = S(b, i, j, IDN); //equilibrium condition on same level
    p(b, i, j) = (gamma - 1.0)*S(b, i, j, IEN);
  });
}


void ReferenceEquilibrium::reset(Mesh& mesh, Real gamma, const View3D& phi, const WBEquilibrium& eq){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int ni = nx1 + 2*ng, nj = nx2 + 2*ng;
  const int is = ng, ie = ng + nx1 - 1, js = ng, je = ng + nx2 - 1;
  View4D U = mesh.get_data_view();
  const View3D rho = rho_, p = p_;
  const WBEquilibrium e = eq;
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj});
  Kokkos::parallel_for("ReferenceResetInterior", policy, KOKKOS_LAMBDA(const int b, const int i, const int j){
    if(i < is || i > ie || j < js || j > je) return;
    const Real r = U(b, i, j, IDN), mx = U(b, i, j, IM1), my = U(b, i, j, IM2); //density and momenta
    rho(b, i, j) = r; //set density
    p(b, i, j) = (gamma - 1.0)*(U(b, i, j, IEN) - 0.5*(mx*mx + my*my)/r); //set pressure including kin energy
  });
  Kokkos::parallel_for("ReferenceResetGhosts", policy, KOKKOS_LAMBDA(const int b, const int i, const int j){
    int si, sj;
    if(ghost_face(i, j, is, ie, js, je, si, sj) < 0) return;
    const Real dphi = phi(b, i, j) - phi(b, si, sj); //potential difference
    rho(b, i, j) = wb_rho0(e, p(b, si, sj), rho(b, si, sj), dphi); //well balancing of density
    p(b, i, j) = wb_p0(e, p(b, si, sj), rho(b, si, sj), dphi);     //and pressure
  });
  sync_same_level_ghosts(mesh, gamma);
  reset_done_ = true;
}

//stores old blocks location, slot, reference array and potential at its centre cell
void ReferenceEquilibrium::begin_regrid(Mesh& mesh, const View3D& phi_old){
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  if(rho_old_.extent(0) == 0){
    rho_old_ = View3D("reference_rho_old", mesh.inputs().max_blocks, ni, nj);
    p_old_ = View3D("reference_p_old", mesh.inputs().max_blocks, ni, nj);
  }
  //just saving the old state into the old private vals
  Kokkos::deep_copy(rho_old_, rho_);
  Kokkos::deep_copy(p_old_, p_);
  old_locs_.clear();
  old_slots_.clear();
  old_phi_c_.clear();
  const int ic = ng + mesh.inputs().nx1_block/2, jc = ng + mesh.inputs().nx2_block/2; //center
  auto h_phi = Kokkos::create_mirror_view(phi_old); //potential
  Kokkos::deep_copy(h_phi, phi_old);   //copy potential
  for(const auto& bp : mesh.blocks()){ //save needed stuff
    old_locs_.push_back(bp->loc);
    old_slots_.push_back(bp->slot_id);
    old_phi_c_.push_back(h_phi(bp->slot_id, ic, jc));
  }
}

//this must know where each new block's reference comes from
void ReferenceEquilibrium::end_regrid(Mesh& mesh, Real gamma, const View3D& phi_new, const WBEquilibrium& eq,
                                      const std::function<void()>& initial_into_next){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  //same as before
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int ni = nx1 + 2*ng, nj = nx2 + 2*ng;
  const int ic = ng + nx1/2, jc = ng + nx2/2;
  const bool use_initial = (eq.type == WB_LINEAR) && !reset_done_;
  Kokkos::View<int*[2], MemorySpace> plan("regrid_plan", n_active);
  Kokkos::View<Real*, MemorySpace> anchor_phi("regrid_anchor_phi", n_active);
  auto h_plan = Kokkos::create_mirror_view(plan); //new regrid plan
  auto h_aphi = Kokkos::create_mirror_view(anchor_phi); //potential
  bool need_initial = false;
  const auto& blocks = mesh.blocks();
  for(int b = 0; b < n_active; ++b){ //iterate thru blocks
    const LogicalLocation loc = blocks[b]->loc; //locations
    int mode = -1, src = -1;
    for(size_t o = 0; o < old_locs_.size() && mode < 0; ++o)
      if(old_locs_[o] == loc){ mode = 0; src = (int)o; } //if same loc
    if(mode < 0 && loc.level > 0){
      const LogicalLocation par = loc.Parent();
      for(size_t o = 0; o < old_locs_.size() && mode < 0; ++o)
        if(old_locs_[o] == par){ mode = 1; src = (int)o; }
    }
    for(size_t o = 0; o < old_locs_.size() && mode < 0; ++o)
      if(old_locs_[o].level > 0 && old_locs_[o].Parent() == loc){ mode = 1; src = (int)o; }
    if(mode < 0) throw std::runtime_error("ReferenceEquilibrium: no old block found for a new block after regrid");
    if(mode == 1 && use_initial){ mode = 2; need_initial = true; }
    h_plan(b, 0) = mode;
    h_plan(b, 1) = old_slots_[src];
    h_aphi(b) = old_phi_c_[src];
  }
  Kokkos::deep_copy(plan, h_plan);
  Kokkos::deep_copy(anchor_phi, h_aphi);
  if(need_initial) initial_into_next();
  View4D S = mesh.get_data_next_view();
  const View3D rho = rho_, p = p_, rho_o = rho_old_, p_o = p_old_;
  const WBEquilibrium e = eq;
  Kokkos::parallel_for("ReferenceRegrid",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const int mode = plan(b, 0), src = plan(b, 1);
      if(mode == 0){
        rho(b, i, j) = rho_o(src, i, j);
        p(b, i, j) = p_o(src, i, j);
      }else if(mode == 1){
        const Real ra = rho_o(src, ic, jc), pa = p_o(src, ic, jc);
        const Real dphi = phi_new(b, i, j) - anchor_phi(b);
        rho(b, i, j) = wb_rho0(e, pa, ra, dphi);
        p(b, i, j) = wb_p0(e, pa, ra, dphi);
      }else{
        const Real r = S(b, i, j, IDN), mx = S(b, i, j, IM1), my = S(b, i, j, IM2);
        rho(b, i, j) = r;
        p(b, i, j) = (gamma - 1.0)*(S(b, i, j, IEN) - 0.5*(mx*mx + my*my)/r);
      }
    });
  sync_same_level_ghosts(mesh, gamma);
}

//here we compare the current state against the initial state as a reference check
Real ReferenceEquilibrium::compare_with_initial(Mesh& mesh, Real gamma, const std::function<void()>& initial_into_next){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 0.0;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  initial_into_next();
  View4D S = mesh.get_data_next_view();
  const View3D rho = rho_, p = p_;
  Real worst = 0.0;
  Kokkos::parallel_reduce("ReferenceCheck",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ng + nx1, ng + nx2}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& w){
      const Real r = S(b, i, j, IDN), mx = S(b, i, j, IM1), my = S(b, i, j, IM2);
      const Real pr = (gamma - 1.0)*(S(b, i, j, IEN) - 0.5*(mx*mx + my*my)/r);
      const Real d = Kokkos::fmax(Kokkos::fabs(rho(b, i, j)/r - 1.0), Kokkos::fabs(p(b, i, j)/pr - 1.0));
      if(d > w) w = d;
    }, Kokkos::Max<Real>(worst)); //worst is the biggest difference
  return worst;
}

//comparison but for ghosts
Real ReferenceEquilibrium::check_reset_ghosts(Mesh& mesh, const View3D& phi, const WBEquilibrium& eq){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 0.0;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int ni = nx1 + 2*ng, nj = nx2 + 2*ng;
  const int is = ng, ie = ng + nx1 - 1, js = ng, je = ng + nx2 - 1;
  const View3D rho = rho_, p = p_;
  const WBEquilibrium e = eq;
  IntView2D n_level = mesh.get_neighbor_level_view();
  Real worst = 0.0;
  Kokkos::parallel_reduce("ReferenceResetGhostCheck",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, ni, nj}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& w){
      int si, sj;
      const int face = ghost_face(i, j, is, ie, js, je, si, sj);
      if(face < 0 || n_level(b, face) == (int)NeighborLevel::Same) return;
      const Real dphi = phi(b, i, j) - phi(b, si, sj);
      const Real r0 = wb_rho0(e, p(b, si, sj), rho(b, si, sj), dphi);
      const Real p0 = wb_p0(e, p(b, si, sj), rho(b, si, sj), dphi);
      const Real d = Kokkos::fmax(Kokkos::fabs(rho(b, i, j)/r0 - 1.0), Kokkos::fabs(p(b, i, j)/p0 - 1.0));
      if(d > w) w = d;
    }, Kokkos::Max<Real>(worst));
  return worst;
}
