#include "mesh.hpp"
#include "slope_limiters.hpp"
#include <stdexcept>
#include <algorithm>
#include <string>


/*
 * Mesh logic and SMR/AMR implementation
 *
 * manages block allocations, quadtree initialization, neighbor mapping,
 * Kokkos device memory views, ghost zone exchanges, and conservative refluxing
*/

//validates domain and block dimensions, computes root block layouts,
//and allocates flat, multi-buffered Kokkos device views for state data and face fluxes
Mesh::Mesh(const MeshInputs& in) : in_(in){
  if(in_.nx1_block <= 0 || in_.nx2_block <= 0)
    throw std::invalid_argument("Mesh: block size must be positive");
  if(in_.domain.nx1 % in_.nx1_block != 0 || in_.domain.nx2 % in_.nx2_block != 0)
    throw std::invalid_argument("Mesh: root grid cell counts must be divisible by MeshBlock size");
  //prolongation/restriction map 2 fine cells onto 1 coarse cell -> block sizes must be even
  if(in_.nx1_block % 2 != 0 || in_.nx2_block % 2 != 0)
    throw std::invalid_argument("Mesh: MeshBlock size must be even for refinement");
  //reconstruction stencil needs 2 ghosts, fine->coarse ghost fill needs 2*nghost fine cells
  if(in_.nghost < 2 || 2*in_.nghost > std::min(in_.nx1_block, in_.nx2_block))
    throw std::invalid_argument("Mesh: need 2 <= nghost <= min(nx1_block, nx2_block)/2");

  nx1_blocks_root_ = in_.domain.nx1/in_.nx1_block;
  nx2_blocks_root_ = in_.domain.nx2/in_.nx2_block;

  int total_cells_i = in_.nx1_block + 2*in_.nghost;
  int total_cells_j = in_.nx2_block + 2*in_.nghost;
  int max_tangent = std::max(in_.nx1_block, in_.nx2_block);

  //initial density reference view for amr condition
  rho_init_view_ = View3D("rho_init_view", in_.max_blocks, total_cells_i, total_cells_j);
  host_rho_init_ = Kokkos::create_mirror_view(rho_init_view_);

  //allocate double-buffered flat device memory pool
  block_dx1_view_ = View1D("block_dx1", in_.max_blocks);
  block_dx2_view_ = View1D("block_dx2", in_.max_blocks);
  host_block_dx1_ = Kokkos::create_mirror_view(block_dx1_view_);
  host_block_dx2_ = Kokkos::create_mirror_view(block_dx2_view_);
  data_view_      = View4D("data_view", in_.max_blocks, total_cells_i, total_cells_j, NVAR);
  data_next_view_ = View4D("data_next_view", in_.max_blocks, total_cells_i, total_cells_j, NVAR);
  flux_view_      = View4D("flux_view", in_.max_blocks, 4, max_tangent, NVAR);
  
  neighbor_gid_   = IntView2D("neighbor_gid", in_.max_blocks, 4);
  neighbor_level_ = IntView2D("neighbor_level", in_.max_blocks, 4);

  host_neighbor_gid_   = Kokkos::create_mirror_view(neighbor_gid_);
  host_neighbor_level_ = Kokkos::create_mirror_view(neighbor_level_);

  //additional bookkeeping views for second finer-neighbor slots and block parity
  neighbor_gid2_ = IntView2D("neighbor_gid2", in_.max_blocks, 4);
  block_parity_  = IntView2D("block_parity",  in_.max_blocks, 2);
  host_neighbor_gid2_ = Kokkos::create_mirror_view(neighbor_gid2_);
  host_block_parity_  = Kokkos::create_mirror_view(block_parity_);
}

//computes physical coordinate boundaries and cell counts for a specific quadtree node location
RegionSize Mesh::BlockSizeFromLocation(const LogicalLocation& loc) const {
  Real dom_w1 = in_.domain.x1max - in_.domain.x1min;
  Real dom_w2 = in_.domain.x2max - in_.domain.x2min;
  std::int64_t n1 = static_cast<std::int64_t>(nx1_blocks_root_) << loc.level;
  std::int64_t n2 = static_cast<std::int64_t>(nx2_blocks_root_) << loc.level;

  RegionSize sz;
  sz.x1min = in_.domain.x1min + dom_w1*(Real)loc.lx1/(Real)n1;
  sz.x1max = in_.domain.x1min + dom_w1*(Real)(loc.lx1 + 1)/(Real)n1;
  sz.x2min = in_.domain.x2min + dom_w2*(Real)loc.lx2/(Real)n2;
  sz.x2max = in_.domain.x2min + dom_w2*(Real)(loc.lx2 + 1)/(Real)n2;
  sz.nx1 = in_.nx1_block;
  sz.nx2 = in_.nx2_block;
  return sz;
}

//constructs root block trees and applies static refinement criteria across defined regions
void Mesh::BuildTree(){
  roots_.clear();
  roots_.reserve((size_t)nx1_blocks_root_*(size_t)nx2_blocks_root_);
  for(int j = 0; j < nx2_blocks_root_; ++j){
    for (int i = 0; i < nx1_blocks_root_; ++i){
      LogicalLocation loc{0, i, j};
      roots_.push_back(std::make_unique<MeshBlockTree>(this, nullptr, loc));
    }
  }

  for(const auto& reg : in_.refine_regions){
    for(auto& root : roots_){
      root->RefineRegion(reg.x1min, reg.x1max, reg.x2min, reg.x2max, reg.level);
    }
  }

  BalanceTree();
}

MeshBlockTree* Mesh::FindNode(const LogicalLocation& loc) const {
  std::int64_t r1 = loc.lx1 >> loc.level;
  std::int64_t r2 = loc.lx2 >> loc.level;
  MeshBlockTree* node = roots_[r2*nx1_blocks_root_ + r1].get(); //roots are stored j-major (see BuildTree)
  for(int l = 1; l <= loc.level && !node->IsLeaf(); ++l){
    int bit1 = (int)((loc.lx1 >> (loc.level - l)) & 1);
    int bit2 = (int)((loc.lx2 >> (loc.level - l)) & 1);
    node = node->Child(bit1 | (bit2 << 1));
  }
  return node;
}

bool Mesh::NeedsBalanceRefinement(const LogicalLocation& loc) const {
  std::int64_t n1 = static_cast<std::int64_t>(nx1_blocks_root_) << loc.level;
  std::int64_t n2 = static_cast<std::int64_t>(nx2_blocks_root_) << loc.level;

  for(int fi = 0; fi < 4; ++fi){
    FaceDir face = static_cast<FaceDir>(fi);
    bool xface = (face == FaceDir::XM || face == FaceDir::XP);
    int shift = (face == FaceDir::XP || face == FaceDir::YP) ? +1 : -1;

    //same-level location across this face, wrapped for periodic directions
    LogicalLocation nb = loc;
    std::int64_t& lx = xface ? nb.lx1 : nb.lx2;
    std::int64_t n = xface ? n1 : n2;
    bool periodic = xface ? in_.periodic_x1 : in_.periodic_x2;
    lx += shift;
    if(lx < 0 || lx >= n){
      if(!periodic) continue; //physical boundary
      lx = (lx + n) % n;
    }

    //a coarser or same-level leaf over there is fine
    MeshBlockTree* node = FindNode(nb);
    if(node->GetLocation().level < loc.level || node->IsLeaf()) continue;

    //same-level node is refined: its two children facing us must be leaves
    for(int q = 0; q < 4; ++q){
      int qx = q & 1, qy = (q >> 1) & 1;
      bool facing = xface ? (qx == (shift > 0 ? 0 : 1)) : (qy == (shift > 0 ? 0 : 1));
      if(facing && !node->Child(q)->IsLeaf()) return true;
    }
  }
  return false;
}

void Mesh::BalanceTree(){
  bool changed = true;
  while(changed){
    changed = false;
    std::vector<LogicalLocation> leaves;
    for(auto& root : roots_) root->CollectLeaves(leaves);
    for(const auto& loc : leaves){
      if(NeedsBalanceRefinement(loc)){
        FindNode(loc)->Refine();
        changed = true;
      }
    }
  }
}


//initializes an individual mesh block, setting up ghost zone offsets and physical boundary flags
MeshBlock::MeshBlock(int gid_, int slot_id_, LogicalLocation loc_, RegionSize size_, int nghost_)
  : gid(gid_), slot_id(slot_id_), loc(loc_), block_size(size_), nghost(nghost_) {
  is = nghost;
  ie = nghost + block_size.nx1 - 1;
  js = nghost;
  je = nghost + block_size.nx2 - 1;
  for(int i = 0; i < 4; ++i){
    is_physical_boundary[i] = true;
  }
}

//collects leaf nodes from the quadtree and allocates active computational mesh blocks
void Mesh::CreateMeshBlocks(){
  std::vector<LogicalLocation> leaves;
  for(auto& root : roots_) root->CollectLeaves(leaves);

  if(leaves.size() > static_cast<size_t>(in_.max_blocks)) {
    throw std::runtime_error("Mesh needs " + std::to_string(leaves.size()) + 
                             " blocks but max_blocks = " + std::to_string(in_.max_blocks) +
                             ": increase max_blocks in the parameter file");
  }

  blocks_.clear();
  blocks_.reserve(leaves.size());
  int gid = 0;
  for(const auto& loc : leaves){
    RegionSize sz = BlockSizeFromLocation(loc);
    blocks_.push_back(std::make_unique<MeshBlock>(gid, gid, loc, sz, in_.nghost));
    ++gid;
  }
}

namespace{
struct IndexRange{ std::int64_t lo, hi; };

IndexRange RangeAtLevel(std::int64_t lx, int level, int common_level){
  std::int64_t scale = 1LL << (common_level - level);
  return { lx * scale, (lx + 1) * scale - 1 };
}

bool Overlap(const IndexRange& a, const IndexRange& b){
  return a.lo <= b.hi && b.lo <= a.hi;
}
} // namespace

//establishes topological neighbor relationships between blocks across multi-level AMR boundaries
void Mesh::SetupNeighbors(){
  for(auto& bptr : blocks_){
    MeshBlock& b = *bptr;
    for(int f = 0; f < 4; ++f){
      b.is_physical_boundary[f] = true;
      b.neighbors[f].clear();
    }
  }

  for(auto& aptr : blocks_){
    MeshBlock& a = *aptr;
    for(int fi = 0; fi < 4; ++fi){
      FaceDir face = static_cast<FaceDir>(fi);
      for(auto& bptr : blocks_){
        MeshBlock& b = *bptr;
        bool same_block = (&a == &b);

        int lc = std::max(a.loc.level, b.loc.level);
        IndexRange a1 = RangeAtLevel(a.loc.lx1, a.loc.level, lc);
        IndexRange a2 = RangeAtLevel(a.loc.lx2, a.loc.level, lc);

        std::int64_t domain_w1 = static_cast<std::int64_t>(nx1_blocks_root_) << lc;
        std::int64_t domain_w2 = static_cast<std::int64_t>(nx2_blocks_root_) << lc;

        for(int wrap = 0; wrap < 2; ++wrap){
          if(wrap == 0 && same_block) continue;
          if(wrap == 1 && !((face == FaceDir::XM || face == FaceDir::XP) ? in_.periodic_x1 : in_.periodic_x2))
              continue;

          IndexRange b1 = RangeAtLevel(b.loc.lx1, b.loc.level, lc);
          IndexRange b2 = RangeAtLevel(b.loc.lx2, b.loc.level, lc);
          if(wrap == 1){
            if(face == FaceDir::XM || face == FaceDir::XP){ b1.lo -= domain_w1; b1.hi -= domain_w1; }
            else{ b2.lo -= domain_w2; b2.hi -= domain_w2; }
          }

          bool adjacent = false;
          switch(face){
            case FaceDir::XP: adjacent = (b1.lo == a1.hi + 1) && Overlap(a2, b2); break;
            case FaceDir::XM: adjacent = (a1.lo == b1.hi + 1) && Overlap(a2, b2); break;
            case FaceDir::YP: adjacent = (b2.lo == a2.hi + 1) && Overlap(a1, b1); break;
            case FaceDir::YM: adjacent = (a2.lo == b2.hi + 1) && Overlap(a1, b1); break;
          }

          if(!adjacent && wrap == 1){
            IndexRange b1b = RangeAtLevel(b.loc.lx1, b.loc.level, lc);
            IndexRange b2b = RangeAtLevel(b.loc.lx2, b.loc.level, lc);
            if(face == FaceDir::XM || face == FaceDir::XP){ b1b.lo += domain_w1; b1b.hi += domain_w1; }
            else{ b2b.lo += domain_w2; b2b.hi += domain_w2; }
            switch(face){
              case FaceDir::XP: adjacent = (b1b.lo == a1.hi + 1) && Overlap(a2, b2b); break;
              case FaceDir::XM: adjacent = (a1.lo == b1b.hi + 1) && Overlap(a2, b2b); break;
              case FaceDir::YP: adjacent = (b2b.lo == a2.hi + 1) && Overlap(a1, b1b); break;
              case FaceDir::YM: adjacent = (a2.lo == b2b.hi + 1) && Overlap(a1, b1b); break;
            }
          }

          if(adjacent){
            NeighborLevel lvl = (b.loc.level == a.loc.level) ? NeighborLevel::Same
                              : (b.loc.level <  a.loc.level) ? NeighborLevel::Coarser
                                                             : NeighborLevel::Finer;
            a.neighbors[fi].push_back(NeighborBlock{b.gid, lvl, wrap == 1});
            a.is_physical_boundary[fi] = false;
          }
        }
      }
    }
  }

  SyncNeighborTablesDevice();
}

//meshblock sync logic uses gid2 for bookkeeping
//synchronizes host neighbor topology maps, secondary finer slots, and parity flags down to Kokkos device views.
void Mesh::SyncNeighborTablesDevice(){
  for(size_t b = 0; b < blocks_.size(); ++b){
    host_block_dx1_(b) = blocks_[b]->dx1();
    host_block_dx2_(b) = blocks_[b]->dx2();
    host_block_parity_(b, 0) = (int)(blocks_[b]->loc.lx1 & 1);
    host_block_parity_(b, 1) = (int)(blocks_[b]->loc.lx2 & 1);

    for(int f = 0; f < 4; ++f){
      host_neighbor_gid2_(b, f) = -1;

      if(blocks_[b]->is_physical_boundary[f]){
        host_neighbor_gid_(b, f)   = -1;
        host_neighbor_level_(b, f) = static_cast<int>(NeighborLevel::PhysicalBoundary);
        continue;
      }
      if(blocks_[b]->neighbors[f].empty()) continue;

      NeighborLevel lvl = blocks_[b]->neighbors[f][0].level;
      host_neighbor_level_(b, f) = static_cast<int>(lvl);

      if(lvl == NeighborLevel::Finer){
        if(blocks_[b]->neighbors[f].size() != 2)
          throw std::runtime_error("SyncNeighborTablesDevice: expected 2 finer neighbors per face (2:1 balance violated)");
        host_neighbor_gid_(b, f) = -1;
        bool xface = (f == (int)FaceDir::XM || f == (int)FaceDir::XP);
        for(const auto& nb : blocks_[b]->neighbors[f]){
          const auto& bloc = blocks_[nb.gid]->loc;
          int half = xface ? (int)(bloc.lx2 & 1) : (int)(bloc.lx1 & 1);
          if(half == 0) host_neighbor_gid_(b, f)  = nb.gid;
          else           host_neighbor_gid2_(b, f) = nb.gid;
        }
      } else {
        host_neighbor_gid_(b, f) = blocks_[b]->neighbors[f][0].gid;
      }
    }
  }
  Kokkos::deep_copy(neighbor_gid_,   host_neighbor_gid_);
  Kokkos::deep_copy(neighbor_level_, host_neighbor_level_);
  Kokkos::deep_copy(neighbor_gid2_,  host_neighbor_gid2_);
  Kokkos::deep_copy(block_parity_,   host_block_parity_);
}

Real MeshBlock::x1v(int i) const {
  return block_size.x1min + (i - is + 0.5) * dx1();
}
Real MeshBlock::x2v(int j) const {
  return block_size.x2min + (j - js + 0.5) * dx2();
}

//parallel Kokkos kernel executing ghost zone data exchanges across Same, Coarser, and Finer block boundaries
void ExchangeGhostZones(Mesh& mesh){
  int n_active = mesh.num_active_blocks();
  if (n_active == 0) return;

  View4D data = mesh.get_data_view();
  IntView2D n_gid = mesh.get_neighbor_gid_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  IntView2D n_gid2 = mesh.get_neighbor_gid2_view();   //gid2 was needed and parity
  IntView2D parity = mesh.get_block_parity_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;

  Kokkos::parallel_for(
    "ExchangeGhostZones",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, std::max(nx1, nx2)}),
    KOKKOS_LAMBDA(const int a_slot, const int fi, const int t){
      int lvl = n_level(a_slot, fi);
      if(lvl == static_cast<int>(NeighborLevel::PhysicalBoundary)) return;

      int b_slot = n_gid(a_slot, fi);
      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = xface ? nx2 : nx1;

      if(t >= N) return;

      if(lvl == static_cast<int>(NeighborLevel::Same)){
        //direct copy for matching resolution neighbors
        if(face == FaceDir::XP){
          for(int g = 0; g < nghost; ++g)
            for(size_t k = 0; k < NVAR; ++k) data(a_slot, ie + 1 + g, js + t, k) = data(b_slot, is + g, js + t, k);
        }else if(face == FaceDir::XM){
          for(int g = 0; g < nghost; ++g)
              for(size_t k = 0; k < NVAR; ++k) data(a_slot, is - 1 - g, js + t, k) = data(b_slot, ie - g, js + t, k);
        }else if(face == FaceDir::YP){
          for(int g = 0; g < nghost; ++g)
            for(size_t k = 0; k < NVAR; ++k) data(a_slot, is + t, je + 1 + g, k) = data(b_slot, is + t, js + g, k);
        }else if(face == FaceDir::YM){
          for(int g = 0; g < nghost; ++g)
            for(size_t k = 0; k < NVAR; ++k) data(a_slot, is + t, js - 1 - g, k) = data(b_slot, is + t, je - g, k);
        }
      } 
      else if(lvl == static_cast<int>(NeighborLevel::Coarser)){
        //slope-limited linear prolongation from coarser neighbor grids
        //fine ghost layers g = 0,1 both lie inside the first coarse cell -> coarse index g/2
        int half_N = N/2;
        int half = xface ? parity(a_slot, 1) : parity(a_slot, 0);
        int tk = half * half_N + t / 2;
        
        //determine coarse-grid coordinate bounds and clamped stencil indices
        int min_idx = xface ? js : is;
        int max_idx = xface ? je : ie;
        int c_idx = min_idx + tk;
        int c_left  = Kokkos::max(min_idx, c_idx - 1);
        int c_right = Kokkos::min(max_idx, c_idx + 1);

        //subcell offset relative to the coarse cell center (-0.25 or +0.25)
        Real sub_offset = (t % 2 == 0) ? -0.25 : 0.25;

        for(int g = 0; g < nghost; ++g){
          if(face == FaceDir::XP){
            int coarse_i = is + g/2;
            for(size_t k = 0; k < NVAR; ++k) {
              Real q_center = data(b_slot, coarse_i, c_idx, k);
              Real q_left   = data(b_slot, coarse_i, c_left, k);
              Real q_right  = data(b_slot, coarse_i, c_right, k);
              Real slope    = minmod(q_center - q_left, q_right - q_center);
              data(a_slot, ie + 1 + g, js + t, k) = q_center + sub_offset * slope;
            }
          }else if(face == FaceDir::XM){
            int coarse_i = ie - g/2;
            for(size_t k = 0; k < NVAR; ++k) {
              Real q_center = data(b_slot, coarse_i, c_idx, k);
              Real q_left   = data(b_slot, coarse_i, c_left, k);
              Real q_right  = data(b_slot, coarse_i, c_right, k);
              Real slope    = minmod(q_center - q_left, q_right - q_center);
              data(a_slot, is - 1 - g, js + t, k) = q_center + sub_offset * slope;
            }
          }else if(face == FaceDir::YP){
            int coarse_j = js + g/2;
            for(size_t k = 0; k < NVAR; ++k) {
              Real q_center = data(b_slot, c_idx, coarse_j, k);
              Real q_left   = data(b_slot, c_left, coarse_j, k);
              Real q_right  = data(b_slot, c_right, coarse_j, k);
              Real slope    = minmod(q_center - q_left, q_right - q_center);
              data(a_slot, is + t, je + 1 + g, k) = q_center + sub_offset * slope;
            }
          }else if(face == FaceDir::YM){
            int coarse_j = je - g/2;
            for(size_t k = 0; k < NVAR; ++k) {
              Real q_center = data(b_slot, c_idx, coarse_j, k);
              Real q_left   = data(b_slot, c_left, coarse_j, k);
              Real q_right  = data(b_slot, c_right, coarse_j, k);
              Real slope    = minmod(q_center - q_left, q_right - q_center);
              data(a_slot, is + t, js - 1 - g, k) = q_center + sub_offset * slope;
            }
          }
        }
      } 
      else if(lvl == static_cast<int>(NeighborLevel::Finer)){
        //averaging restriction from finer neighbor blocks
        int half_N = N/2;
        int half = t/half_N;
        int local_t = t%half_N;
        int tb = 2*local_t;
        int b_slot_f = (half == 0) ? b_slot : n_gid2(a_slot, fi);

        //each coarse ghost cell covers 2x2 fine cells: normal indices 2g, 2g+1 and tangential tb, tb+1
        for(int g = 0; g < nghost; ++g){
          int fn0 = 2*g, fn1 = 2*g + 1;
          if(face == FaceDir::XP){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, ie + 1 + g, js + t, k) = 0.25*(data(b_slot_f, is + fn0, js + tb, k) + data(b_slot_f, is + fn0, js + tb + 1, k) +
                                                          data(b_slot_f, is + fn1, js + tb, k) + data(b_slot_f, is + fn1, js + tb + 1, k));
          }else if(face == FaceDir::XM){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is - 1 - g, js + t, k) = 0.25*(data(b_slot_f, ie - fn0, js + tb, k) + data(b_slot_f, ie - fn0, js + tb + 1, k) +
                                                          data(b_slot_f, ie - fn1, js + tb, k) + data(b_slot_f, ie - fn1, js + tb + 1, k));
          }else if(face == FaceDir::YP){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is + t, je + 1 + g, k) = 0.25*(data(b_slot_f, is + tb, js + fn0, k) + data(b_slot_f, is + tb + 1, js + fn0, k) +
                                                          data(b_slot_f, is + tb, js + fn1, k) + data(b_slot_f, is + tb + 1, js + fn1, k));
          }else if(face == FaceDir::YM){
            for(size_t k = 0; k < NVAR; ++k)
              data(a_slot, is + t, js - 1 - g, k) = 0.25*(data(b_slot_f, is + tb, je - fn0, k) + data(b_slot_f, is + tb + 1, je - fn0, k) +
                                                          data(b_slot_f, is + tb, je - fn1, k) + data(b_slot_f, is + tb + 1, je - fn1, k));
          }
        }
      }
    }
  );
}

//applies conservative refluxing corrections across fine-coarse block interfaces to ensure mass/momentum conservation
void ApplyReflux(Mesh& mesh, Real dt){
  int n_active = mesh.num_active_blocks();
  if (n_active == 0) return;

  View4D data = mesh.get_data_view();
  View4D flux = mesh.get_flux_view();
  IntView2D n_gid = mesh.get_neighbor_gid_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  IntView2D n_gid2 = mesh.get_neighbor_gid2_view();
  IntView2D parity = mesh.get_block_parity_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  //block-local spacing views
  View1D dx1_view = mesh.get_block_dx1_view();
  View1D dx2_view = mesh.get_block_dx2_view();

  Kokkos::parallel_for(
    "ApplyReflux",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, std::max(nx1, nx2)}),
    KOKKOS_LAMBDA(const int a_slot, const int fi, const int t_idx) {
      if(n_level(a_slot, fi) != static_cast<int>(NeighborLevel::Finer)) return;

      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = xface ? nx2 : nx1;
      if (t_idx >= N) return;

      int half_N = N/2;
      int half = t_idx/half_N;
      int local_t = t_idx%half_N;
      int tb = 2*local_t;
      int b_slot = (half == 0) ? n_gid(a_slot, fi) : n_gid2(a_slot, fi);

      Real d = xface ? dx1_view(a_slot) : dx2_view(a_slot);
      Real sign = (face == FaceDir::XP || face == FaceDir::YP) ? +1.0 : -1.0;
      int opp = xface ? (fi == (int)FaceDir::XP ? (int)FaceDir::XM : (int)FaceDir::XP)
                      : (fi == (int)FaceDir::YP ? (int)FaceDir::YM : (int)FaceDir::YP);

      int is = nghost, ie = nghost + nx1 - 1;
      int js = nghost, je = nghost + nx2 - 1;

      size_t ai = (face == FaceDir::XP) ? ie : (face == FaceDir::XM) ? is : (is + t_idx);
      size_t aj = (face == FaceDir::YP) ? je : (face == FaceDir::YM) ? js : (js + t_idx);

      for(size_t k = 0; k < NVAR; ++k){
        Real f_eff = 0.5*(flux(b_slot, opp, tb, k) + flux(b_slot, opp, tb + 1, k));
        //corner cells get corrected by two faces at once -> atomic
        Kokkos::atomic_add(&data(a_slot, ai, aj, k), sign*(dt/d)*(flux(a_slot, fi, t_idx, k) - f_eff));
      }
    }
  );
}

void Mesh::UpdateDynamicMesh(const BlockCriterion& refine_crit,
                             const BlockCriterion& derefine_crit,
                             int max_level, Real derefine_treshold){
  for(auto& root : roots_){
    root->EvaluateDynamicRefinement(refine_crit, data_view_, rho_init_view_, max_level);
  }

  for(auto& root : roots_){
    root->EvaluateDynamicDerefinement(derefine_crit, data_view_, rho_init_view_, derefine_treshold);
  }

  //re-refines anything the passes above left unbalanced (also undoes derefinements that would break 2:1)
  BalanceTree();

  CreateMeshBlocks();
  SetupNeighbors();
}

void Mesh::CaptureInitialState(){
  int n_active = num_active_blocks();
  int nx1 = in_.nx1_block;
  int nx2 = in_.nx2_block;
  int nghost = in_.nghost;

  View4D data = data_view_;
  View3D rho_init = rho_init_view_;

  Kokkos::parallel_for("CaptureInitialState",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, nx1 + 2*nghost, nx2 + 2*nghost}),
    KOKKOS_LAMBDA(int b, int i, int j){
      rho_init(b, i, j) = data(b, i, j, 0); //density is variable index 0
    }
  );
  Kokkos::fence();
}


void Mesh::TransferSolution(const std::vector<std::unique_ptr<MeshBlock>>& old_blocks, 
                            const View4D& old_data){
  View4D new_data = data_view_;
  int nghost = in_.nghost;
  int nx1 = in_.nx1_block;
  int nx2 = in_.nx2_block;

  //helper lambda to find an old block by its logical location
  auto find_old_block = [&](const LogicalLocation& target_loc) -> int{
    for(size_t i = 0; i < old_blocks.size(); ++i){
      if(old_blocks[i]->loc == target_loc){
        return static_cast<int>(i);
      }
    }
    return -1;
  };

  //loop over all newly created active blocks
  for(size_t new_slot = 0; new_slot < blocks_.size(); ++new_slot){
    MeshBlock& new_b = *blocks_[new_slot];
    LogicalLocation new_loc = new_b.loc;

    //case 1: exact match (block didn't change level or position)
    int old_slot = find_old_block(new_loc);
    if(old_slot != -1){
      Kokkos::parallel_for(
        "TransferDirect",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx1 + 2*nghost, nx2 + 2*nghost}),
        KOKKOS_LAMBDA(int i, int j){
          for (size_t k = 0; k < NVAR; ++k){
            new_data(new_slot, i, j, k) = old_data(old_slot, i, j, k);
          }
        }
      );
      continue;
    }

    //case 2: refinement / prolongation (parent exists in old blocks, new block is a child)
    LogicalLocation parent_loc = new_loc.Parent();
    int old_parent_slot = find_old_block(parent_loc);
    if(old_parent_slot != -1){
      //determine child quadrant parity/offsets based on lx1 and lx2
      int half_x1 = new_loc.lx1 & 1; // 0 = left half, 1 = right half
      int half_x2 = new_loc.lx2 & 1; // 0 = bottom half, 1 = top half

      Kokkos::parallel_for(
        "TransferProlongation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost}, {nghost + nx1, nghost + nx2}),
        KOKKOS_LAMBDA(int i, int j) {
          //map local fine cell (i, j) to coarse parent cell coordinates
          int local_i = i - nghost;
          int local_j = j - nghost;

          int coarse_local_i = half_x1*(nx1/2) + (local_i/2);
          int coarse_local_j = half_x2*(nx2/2) + (local_j/2);
          int coarse_i = nghost + coarse_local_i;
          int coarse_j = nghost + coarse_local_j;

          //minmod-limited linear prolongation (same style as the ghost zones), fine cell centers sit at
          //+-0.25 coarse cells -> the 4 children average back to the parent value (conservative).
          //coarse_i +- 1 may be a parent ghost cell, those were filled before old_data was copied
          Real sub_offset_i = (local_i%2 == 0) ? -0.25 : 0.25;
          Real sub_offset_j = (local_j%2 == 0) ? -0.25 : 0.25;

          for (size_t k = 0; k < NVAR; ++k) {
            Real q_center = old_data(old_parent_slot, coarse_i, coarse_j, k);
            Real slope_i = minmod(q_center - old_data(old_parent_slot, coarse_i - 1, coarse_j, k),
                                  old_data(old_parent_slot, coarse_i + 1, coarse_j, k) - q_center);
            Real slope_j = minmod(q_center - old_data(old_parent_slot, coarse_i, coarse_j - 1, k),
                                  old_data(old_parent_slot, coarse_i, coarse_j + 1, k) - q_center);
            new_data(new_slot, i, j, k) = q_center + sub_offset_i*slope_i + sub_offset_j*slope_j;
          }
        }
      );
      continue;
    }

    //case 3: derefinement / restriction (4 children merge into this new coarse block)
    //find all 4 children in old blocks whose parent is new_loc:
    bool found_all_children = true;
    int old_child_slots[4];
    for(int q = 0; q < 4; ++q){
      std::int64_t clx1 = (new_loc.lx1 << 1) | (q & 1);
      std::int64_t clx2 = (new_loc.lx2 << 1) | ((q >> 1) & 1);
      LogicalLocation child_loc{new_loc.level + 1, clx1, clx2};
      old_child_slots[q] = find_old_block(child_loc);
      if(old_child_slots[q] == -1){
        found_all_children = false;
        break;
      }
    }

    if(found_all_children){
      Kokkos::parallel_for("TransferRestriction",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost}, {nghost + nx1, nghost + nx2}),
        KOKKOS_LAMBDA(int i, int j){
          int local_i = i - nghost;
          int local_j = j - nghost;

          //determines wich child quartet this coarse cell maps to and the fine indices
          int q_x = (local_i >= nx1/2 ? 1 : 0);
          int q_y = (local_j >= nx2/2 ? 1 : 0);
          int q = q_x + 2 * q_y;
          int fine_local_i = 2*(local_i%(nx1/2));
          int fine_local_j = 2*(local_j%(nx2/2));
          int fine_i = nghost + fine_local_i;
          int fine_j = nghost + fine_local_j;

          int child_slot = old_child_slots[q];

          for(size_t k = 0; k < NVAR; ++k){
            //avg 2x2 fine cells into single coarse cell
            Real val00 = old_data(child_slot, fine_i,     fine_j,     k);
            Real val10 = old_data(child_slot, fine_i + 1, fine_j,     k);
            Real val01 = old_data(child_slot, fine_i,     fine_j + 1, k);
            Real val11 = old_data(child_slot, fine_i + 1, fine_j + 1, k);

            new_data(new_slot, i, j, k) = 0.25*(val00 + val01 + val10 + val11);
          }
        }
      );
      continue;
    }
  }
}
