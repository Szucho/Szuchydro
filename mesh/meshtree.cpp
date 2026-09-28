#include "mesh.hpp"
#include <iterator>

/*
 * Adaptive Mesh Refinement (AMR) quadtree implementation
 * 
 * manages recursive quadtree node subdivision, bounding box intersection tests
 * for targeted refinement regions, and leaf collection
*/


//constructs an individual quadtree node with a given logical location, parent link, and mesh context
MeshBlockTree::MeshBlockTree(Mesh* mesh, MeshBlockTree* parent, LogicalLocation loc)
  : mesh_(mesh), parent_(parent), loc_(loc) {}

//subdivides a leaf node into 4 child quadrants (quadtree children),
//computing their refined logical locations using bitwise indexing
void MeshBlockTree::Refine(){
  if(!IsLeaf()) return;
  for(int q = 0; q < 4; ++q){
    std::int64_t clx1 = (loc_.lx1 << 1) | (q & 1);
    std::int64_t clx2 = (loc_.lx2 << 1) | ((q >> 1) & 1);
    LogicalLocation cloc{loc_.level + 1, clx1, clx2};
    children_[q] = std::make_unique<MeshBlockTree>(mesh_, this, cloc);
  }
}

//removes and deallocates all child nodes, reverting this node back to a leaf
void MeshBlockTree::Derefine(){
  for (auto& c : children_) c.reset();
}

//recursively evaluates spatial intersection between the current node's bounding box
//and a specified refinement region, triggering subdivision if the target level is not yet met
void MeshBlockTree::RefineRegion(const RegionSize& root_domain,
                               int nx1_blocks_root, int nx2_blocks_root,
                               double x1min, double x1max,
                               double x2min, double x2max,
                               int target_level) {

  if(loc_.level >= target_level) return;

  //calculate physical coordinate boundaries for the current tree node
  double dom_w1 = root_domain.x1max - root_domain.x1min;
  double dom_w2 = root_domain.x2max - root_domain.x2min;
  std::int64_t n1_at_level = static_cast<std::int64_t>(nx1_blocks_root) << loc_.level;
  std::int64_t n2_at_level = static_cast<std::int64_t>(nx2_blocks_root) << loc_.level;
  double bx1min = root_domain.x1min + dom_w1*(double)loc_.lx1/(double)n1_at_level;
  double bx1max = root_domain.x1min + dom_w1*(double)(loc_.lx1 + 1)/(double)n1_at_level;
  double bx2min = root_domain.x2min + dom_w2*(double)loc_.lx2/(double)n2_at_level;
  double bx2max = root_domain.x2min + dom_w2*(double)(loc_.lx2 + 1)/(double)n2_at_level;

  //check for spatial bounding box overlap with the requested refinement zone
  bool overlaps = (bx1min < x1max) && (bx1max > x1min) &&
                  (bx2min < x2max) && (bx2max > x2min);
  if(!overlaps) return;

  //if overlapping and currently a leaf, perform refinement and recurse into children
  if(IsLeaf()) Refine();
  for (auto& c : children_){
    c->RefineRegion(root_domain, nx1_blocks_root, nx2_blocks_root,
                    x1min, x1max, x2min, x2max, target_level);
  }
}

//recursively traverses the quadtree to gather logical locations of all active leaf nodes
void MeshBlockTree::CollectLeaves(std::vector<LogicalLocation>& out) const {
  if(IsLeaf()){
    out.push_back(loc_);
    return;
  }
  for(const auto& c : children_) c->CollectLeaves(out);
}

//recursively evaluates a dynamic refinement criterion across the quadtree
void MeshBlockTree::EvaluateDynamicRefinement(const RefinementCriterion& criterion, const View4D& data,
                                              const View3D& rho_init, int max_level){
  if(!IsLeaf()){
    //if it's already an internal node -> recurse into children
    for(auto& c : children_){
      c->EvaluateDynamicRefinement(criterion, data, rho_init, max_level);
    }
    return;
  }

  //if it IS a leaf node we check if we can refine further
  if(loc_.level >= max_level) return;

  //find the actual active block slot corresponding to this leaf node's location
  int actual_slot_id = -1;
  for(const auto& bptr : mesh_->blocks()){
    if(bptr->loc == loc_){
      actual_slot_id = bptr->slot_id;
      break;
    }
  }
  if(actual_slot_id == -1) return; // Leaf not active yet
  //
  RegionSize sz = mesh_->BlockSizeFromLocation(loc_);
  MeshBlock temp_block(actual_slot_id, actual_slot_id, loc_, sz, mesh_->inputs().nghost);

  //user defined condition function evaluation
  if(criterion(temp_block, data, rho_init)){
    Refine();
    //recurse into the newly created children to continue eval if needed
    for(auto& c : children_){
      c->EvaluateDynamicRefinement(criterion, data, rho_init, max_level);
    }
  }
}

//evaluates derefinement bottom-up across the quadtree
void MeshBlockTree::EvaluateDynamicDerefinement(const DerefinementCriterion& criterion, 
                                                const View4D& data, const View3D& rho_init,
                                                Real derefine_threshold) {
  if (IsLeaf()) return; //leaf nodes cannot be derefined

  //recurse down to the deepest branches first (bottom-up traversal)
  for (auto& c : children_) {
    if (!c->IsLeaf()) {
      c->EvaluateDynamicDerefinement(criterion, data, rho_init, derefine_threshold);
    }
  }

  //check if all four children are now leaves
  bool all_children_leaves = true;
  for(auto& c : children_){
    if(!c->IsLeaf()){
      all_children_leaves = false;
      break;
    }
  }

  if(all_children_leaves){
    //evaluate if ALL four children fall below the derefinement threshold
    bool safe_to_derefine = true;
    for(auto& c : children_){
      //finding actual active block slot for this child
      int child_slot_id = -1;
      for(const auto& bptr : mesh_->blocks()){
        if(bptr->loc == c->GetLocation()){
          child_slot_id = bptr->slot_id;
          break;
        }
      }
      if(child_slot_id == -1){
        safe_to_derefine = false;
        break;
      }
      RegionSize sz = mesh_->BlockSizeFromLocation(c->GetLocation());
      MeshBlock temp_block(child_slot_id, child_slot_id, c->GetLocation(), sz, mesh_->inputs().nghost);
      
      //If even ONE child still exceeds the derefine threshold, do NOT derefine
      if(criterion(temp_block, data, rho_init)){
        safe_to_derefine = false;
        break;
      }
    }

    if(safe_to_derefine){
      Derefine(); //collapse the 4 children back into this parent node
    }
  }
}
