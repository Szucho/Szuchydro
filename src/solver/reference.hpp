#pragma once

#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"

/*
 Reference equilibrium for well-balancing across refinement levels
 This is sort of a patchwork fix and maybe I will have to return here to fix the whole thing T-T

 At a level boundary the ghost exchange interpolates coarse data onto fine ghost cells
 (prolongation) and averages fine data onto coarse ghost cells (restriction). Neither operation
 knows the hydrostatic equilibrium, so it creates ghost cells that deviate from the discrete
 equilibrium of their level, and the well-balanced scheme turns that deviation into spurious flow.
 This defeats the whole purpose of the well-balanced method

 Fix: run the exchange on the DEVIATION from a reference equilibrium of each level,
   delta = U - U_ref,   U_ref = (rho_ref, 0, 0, p_ref/(gamma - 1)),
 and add the reference back afterwards. At rest delta = 0 everywhere, so every ghost cell becomes
 exactly its own level's equilibrium. The round trip (U - U_ref) + U_ref is exact in floating point
 for states within a factor of 2 of the reference (Sterbenz' lemma).

 The reference is captured from the initial state, right after the problem setup: the problem
 fills every block (ghost cells included) at the block's own resolution, so every level gets its
 own exact discrete equilibrium. On faces between blocks of the SAME level (incl. periodic wraps)
 the reference ghost cells are then made exact copies of the neighbour's reference, which keeps
 the exchange conservative (see capture). Static refinement only for now: blocks created later by AMR
 would need a reference built on the fly (M3b.5).
*/
class ReferenceEquilibrium{
public:
  //rho and p of every cell (ghost cells included) of the current state
  void capture(Mesh& mesh, Real gamma);
  //U -> U - U_ref, and back (all cells of the active blocks)
  void to_deviation(Mesh& mesh, Real gamma) const;
  void from_deviation(Mesh& mesh, Real gamma) const;
  bool captured() const { return rho_.extent(0) > 0; }

private:
  View3D rho_, p_;
};
