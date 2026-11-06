#pragma once

#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "hydro/reconstruction.hpp"

/*
 The hydrodynamic right-hand side on the mesh, in two steps:

   ComputeFluxes:        one thread per FACE: reconstruct the left/right states at that face and
                         call the Riemann solver once. Faces on block edges also go into the mesh's
                         flux register, which ApplyReflux uses to correct coarse/fine faces.
   ApplyFluxDivergence:  one thread per CELL: U_to = U_from - dt/dx (Fx[i+1/2] - Fx[i-1/2])
                                                         - dt/dy (Fy[j+1/2] - Fy[j-1/2])

 Before, every cell computed all four of its face fluxes,
 so every interior face was solved twice (once by each neighbour -- T-T my mistake)
 Computing each face once halves the Riemann solves, with exactly the
 same arithmetic, so the results are bit-for-bit unchanged.

 Splitting fluxes from the update is what a multi-stage integrator (e.g. VL2) needs:
 fluxes from the stage state, update applied to the state at the start of the step.
*/

//face fluxes of all blocks:
//  fx(block, f, j, var): x-faces, f = 0..nx1 (f = 0 is the left edge of the block), j = 0..nx2-1
//  fy(block, i, f, var): y-faces, f = 0..nx2 (f = 0 is the bottom edge of the block), i = 0..nx1-1
struct FaceFluxes{
  View4D fx, fy;
};

FaceFluxes AllocateFaceFluxes(const MeshInputs& in);
void ComputeFluxes(Mesh& mesh, const View4D& U, Real gamma, const ReconOptions& recon,
                   const FaceFluxes& F);

void ApplyFluxDivergence(Mesh& mesh, const View4D& U_from, const View4D& U_to,
                         const FaceFluxes& F, Real dt);
