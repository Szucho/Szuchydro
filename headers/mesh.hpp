#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <array>
#include <functional>
#include "hydro_config.hpp"

/*
 * Adaptive Mesh Refinement (AMR) and topology header
 *
 * Defines logical coordinate structures, quadtree nodes, individual mesh blocks,
 * and the global mesh container integrated with Kokkos multi-dimensional views.
 *
 * Based on Athena++
*/

//identifies a block's hierarchical tree level and discrete index coordinates
struct LogicalLocation{
  int level = 0;
  std::int64_t lx1 = 0;
  std::int64_t lx2 = 0;

  //computes the parent location in the quadtree hierarchy
  LogicalLocation Parent() const {
    return LogicalLocation{level - 1, lx1 >> 1, lx2 >> 1};
  }

  bool operator==(const LogicalLocation& o) const {
    return level == o.level && lx1 == o.lx1 && lx2 == o.lx2;
  }

  bool operator<(const LogicalLocation& o) const {
    if(level != o.level) return level < o.level;
    if(lx1 != o.lx1) return lx1 < o.lx1;
    return lx2 < o.lx2;
  }
};

//defines physical coordinate bounds and cell dimensions for a grid domain or block
struct RegionSize{
  Real x1min = 0.0, x1max = 0.0;
  Real x2min = 0.0, x2max = 0.0;
  int nx1 = 0;
  int nx2 = 0;
};

//bounding box descriptor used to apply static refinement regions during tree generation
struct RefinementRegion{
  Real x1min, x1max, x2min, x2max;
  int level;
};

class Mesh;

//face direction enumerations (X-minus, X-plus, Y-minus, Y-plus)
enum class FaceDir{ XM = 0, XP = 1, YM = 2, YP = 3 };
//neighbor relationship levels for ghost zone exchange and refluxing consistency
enum class NeighborLevel{ Same = 0, Coarser = 1, Finer = 2, PhysicalBoundary = 3 };

//describes an adjacent block connection across a specific face boundary
struct NeighborBlock{
  int gid;
  NeighborLevel level;
  bool periodic_wrap;
};


//represents an active computational block containing local cell data (=local grid) and physical boundaries
class MeshBlock{
public:
  MeshBlock(int gid_, int slot_id_, LogicalLocation loc_, RegionSize size_, int nghost_);

  int gid;
  int slot_id;
  LogicalLocation loc;
  RegionSize block_size;
  int nghost;

  //active cell index bounds (excluding ghost zones)
  int is, ie, js, je;

  bool is_physical_boundary[4];
  std::vector<NeighborBlock> neighbors[4];

  //helper methods to calculate physical cell-centered coordinates
  Real x1v(int i) const;
  Real x2v(int j) const;

  //local grid spacing calculations
  Real dx1() const { return (block_size.x1max - block_size.x1min)/block_size.nx1; }
  Real dx2() const { return (block_size.x2max - block_size.x2min)/block_size.nx2; }

  int tangentCount(int face) const {
    return (face == (int)FaceDir::XM || face == (int)FaceDir::XP) ? block_size.nx2 : block_size.nx1;
  }
};

//FOR AMR: per-block criterion
//refinement:   returns true if the block should be refined
//derefinement: returns true if the block is quiet enough to be merged
using BlockCriterion = std::function<bool(const MeshBlock& block, const View4D& data, const View3D& rho_init)>;

//quadtree node class managing recursive domain subdivision and leaf collection for SMR/AMR
class MeshBlockTree{
public:
  MeshBlockTree(Mesh* mesh, MeshBlockTree* parent, LogicalLocation loc);

  bool IsLeaf() const { return children_[0] == nullptr; }
  const LogicalLocation& GetLocation() const { return loc_; }
  MeshBlockTree* Child(int q) const { return children_[q].get(); }

  void Refine();
  void Derefine();
  void RefineRegion(Real x1min, Real x1max,
                    Real x2min, Real x2max,
                    int target_level);
  void CollectLeaves(std::vector<LogicalLocation>& out) const;
  //AMR part: dynamic eval hook
  void EvaluateDynamicRefinement(const BlockCriterion& criterion, const View4D& data,
                                 const View3D& rho_init, int max_level);
  void EvaluateDynamicDerefinement(const BlockCriterion& criterion, const View4D& data,
                                   const View3D& rho_init, Real derefine_threshold);

private:
  Mesh* mesh_;
  MeshBlockTree* parent_;
  LogicalLocation loc_;
  std::array<std::unique_ptr<MeshBlockTree>, 4> children_; //quadtree quadrants
};


//initialization structure holding global domain layout parameters and configuration inputs
struct MeshInputs {
    RegionSize domain;
    int nx1_block = 0, nx2_block = 0;
    int nghost = 2;
    int max_blocks = 256;
    bool periodic_x1 = false, periodic_x2 = false;
    std::vector<RefinementRegion> refine_regions;
};

//big mesh manager handling block allocation, Kokkos device views, and inter-block communication
class Mesh{
public:
  explicit Mesh(const MeshInputs& in);

  void BuildTree();
  //refines leaves until face-neighbor levels differ by at most 1 (2:1 balance)
  void BalanceTree();
  
  //dynamic AMR update wrapper
  void UpdateDynamicMesh(const BlockCriterion& refine_crit,
                         const BlockCriterion& derefine_crit,
                         int max_level, Real derefine_threshold);
  void CaptureInitialState();
  View3D get_rho_init_view() { return rho_init_view_; };
  void TransferSolution(const std::vector<std::unique_ptr<MeshBlock>>& old_blocks, 
                        const View4D& old_data);

  void CreateMeshBlocks();
  void SetupNeighbors();
  void SyncNeighborTablesDevice();
  void swap_data_views(){ std::swap(data_view_, data_next_view_); }

  std::vector<std::unique_ptr<MeshBlock>>& blocks(){ return blocks_; }
  const std::vector<std::unique_ptr<MeshBlock>>& blocks() const { return blocks_; }

  //Kokkos Device View accessors for solvers and kernels
  View1D get_block_dx1_view(){ return block_dx1_view_; }
  View1D get_block_dx2_view(){ return block_dx2_view_;}
  View4D get_data_view(){ return data_view_; }
  View4D get_data_next_view(){ return data_next_view_; }
  View4D get_flux_view(){ return flux_view_; }
  IntView2D get_neighbor_gid_view(){ return neighbor_gid_; }
  IntView2D get_neighbor_level_view(){ return neighbor_level_; }

  IntView2D get_neighbor_gid2_view(){ return neighbor_gid2_; }
  IntView2D get_block_parity_view(){ return block_parity_; }

  const MeshInputs& inputs() const { return in_; }
  int nx1_blocks_root() const { return nx1_blocks_root_; }
  int nx2_blocks_root() const { return nx2_blocks_root_; }
  int num_active_blocks() const { return static_cast<int>(blocks_.size()); }

private:
  MeshInputs in_;
  int nx1_blocks_root_, nx2_blocks_root_;
  std::vector<std::unique_ptr<MeshBlockTree>> roots_;
  std::vector<std::unique_ptr<MeshBlock>> blocks_;

  //double-buffered device memory pool & tables
  View1D block_dx1_view_;
  View1D block_dx2_view_;
  View4D data_view_;
  View4D data_next_view_;
  View4D flux_view_;
  IntView2D neighbor_gid_;
  IntView2D neighbor_level_;
  IntView2D neighbor_gid2_; //second finer-neighbor slot, shape (max_blocks,4), -1 if unused
  IntView2D block_parity_;

  //host mirror tables used to sync connectivity maps down to device views
  HostView1D host_block_dx1_;
  HostView1D host_block_dx2_;
  HostIntView2D host_neighbor_gid_;
  HostIntView2D host_neighbor_level_;
  HostIntView2D host_neighbor_gid2_;
  HostIntView2D host_block_parity_;

  //for the criterion
  View3D rho_init_view_; //stores initial density field [max_blocks, total_cells_i, total_cells_j]
  HostView3D host_rho_init_;

  RegionSize BlockSizeFromLocation(const LogicalLocation& loc) const;
  //deepest existing tree node containing loc (may be a coarser leaf)
  MeshBlockTree* FindNode(const LogicalLocation& loc) const;
  //true if a face neighbor of the leaf at loc is 2+ levels finer
  bool NeedsBalanceRefinement(const LogicalLocation& loc) const;

  friend class MeshBlockTree;
};

void ApplyPhysicalBoundaryConditions(Mesh& mesh, const GridBC& bc);
void ExchangeGhostZones(Mesh& mesh);
void ApplyReflux(Mesh& mesh, Real dt);
