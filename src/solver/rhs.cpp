#include "solver/rhs.hpp"
#include "core/hydro_config.hpp"
#include "hydro/hllc.hpp"
#include "hydro/reconstruction.hpp"

//allocate the fluxes in x, y direcionts
FaceFluxes AllocateFaceFluxes(const MeshInputs& in){
  FaceFluxes F;
  F.fx = View4D("face_flux_x", in.max_blocks, in.nx1_block + 1, in.nx2_block, NVAR);
  F.fy = View4D("face_flux_y", in.max_blocks, in.nx1_block, in.nx2_block + 1, NVAR);
  return F;
}

//then compute them
void ComputeFluxes(Mesh& mesh, const View4D& U, Real gamma, const ReconOptions& recon,
                   const FaceFluxes& F, const View3D& phi, 
                   bool well_balanced, bool first_order){
  const int n_active = mesh.num_active_blocks(); //active meshblocks
  if(n_active == 0) return;
  const int nx1 = mesh.inputs().nx1_block; //nx1, nx2 and ng are local to meshblocks
  const int nx2 = mesh.inputs().nx2_block;
  const int ng  = mesh.inputs().nghost;
  const int is = ng, js = ng;

  View4D flux_face = mesh.get_flux_view(); //block-edge fluxes for refluxing
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  const View4D fx = F.fx, fy = F.fy;
  const ReconOptions opt = recon; //plain struct copied into the kernels
  const bool wb = well_balanced && phi.extent(0) > 0;
  const bool fo = first_order;

  //x-faces: face f lies between cells i-1 and i, with i = is + f
  Kokkos::parallel_for("FluxesX",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, nx1 + 1, nx2}),
    KOKKOS_LAMBDA(const int b, const int f, const int jj){
      const int i = is + f, j = js + jj;
      Vec4 m2, m1, c0, p1;
      for(size_t k = 0; k < NVAR; ++k){ //filling all m2, m1, c0, p1 state vecs
        m2[k] = U(b, i - 2, j, k);
        m1[k] = U(b, i - 1, j, k);
        c0[k] = U(b, i,     j, k);
        p1[k] = U(b, i + 1, j, k);
      }
      Vec4 l, r; //left and right init
      if(fo){ //first order (VL2 predictor)
        if(wb) reconstruct_face_wb_first_order(m1, c0, phi(b, i-1, j), phi(b, i, j), gamma, l, r);
        else reconstruct_face_first_order(m1, c0, l, r);
      
      } else if(wb){
        reconstruct_face_wb(m2, m1, c0, p1, phi(b, i-2, j), phi(b, i-1, j), phi(b, i, j),
                            phi(b, i+1, j), block_dx1(b), gamma, opt.limiter, l, r);
      } else {
        reconstruct_face(m2, m1, c0, p1, block_dx1(b), gamma, opt, l, r);
      }
      const Vec4 flux = FluxhllcX(l, r, gamma); //flux vals
      for(size_t k = 0; k < NVAR; ++k) fx(b, f, jj, k) = flux[k];
      if(f == 0)
        for(size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::XM, jj, k) = flux[k];
      if(f == nx1)
        for(size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::XP, jj, k) = flux[k];
    }
  );

  //y-faces: face f lies between cells j-1 and j, with j = js + f
  //same logic as before
  Kokkos::parallel_for("FluxesY",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, nx1, nx2 + 1}),
    KOKKOS_LAMBDA(const int b, const int ii, const int f){
      const int i = is + ii, j = js + f;
      Vec4 m2, m1, c0, p1;
      for(size_t k = 0; k < NVAR; ++k){
        m2[k] = U(b, i, j - 2, k);
        m1[k] = U(b, i, j - 1, k);
        c0[k] = U(b, i, j,     k);
        p1[k] = U(b, i, j + 1, k);
      }
      Vec4 l, r;
      if(fo){ //first odrer (VL2 predictor)
        if(wb) reconstruct_face_wb_first_order(m1, c0, phi(b, i, j-1), phi(b, i, j), gamma, l, r);
        else reconstruct_face_first_order(m1, c0, l, r);
      } else if(wb){
        reconstruct_face_wb(m2, m1, c0, p1, phi(b, i, j-2), phi(b, i, j-1), phi(b, i, j),
                            phi(b, i, j+1), block_dx2(b), gamma, opt.limiter, l, r);
      } else {
        reconstruct_face(m2, m1, c0, p1, block_dx2(b), gamma, opt, l, r);
      }
      const Vec4 flux = FluxhllcY(l, r, gamma);
      for(size_t k = 0; k < NVAR; ++k) fy(b, ii, f, k) = flux[k];
      if(f == 0)
        for(size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::YM, ii, k) = flux[k];
      if(f == nx2)
        for(size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::YP, ii, k) = flux[k];
    }
  );
}


void ApplyFluxDivergence(Mesh& mesh, const View4D& U_from, const View4D& U_to,
                         const FaceFluxes& F, Real dt){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int nx1 = mesh.inputs().nx1_block;
  const int nx2 = mesh.inputs().nx2_block;
  const int ng  = mesh.inputs().nghost;
  const int is = ng, ie = ng + nx1 - 1;
  const int js = ng, je = ng + nx2 - 1;
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  const View4D fx = F.fx, fy = F.fy;

  Kokkos::parallel_for("FluxDivergence",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const Real dx = block_dx1(b), dy = block_dx2(b);
      const int fi = i - is, fj = j - js; //left / bottom face of this cell
      for(size_t k = 0; k < NVAR; ++k){ //update state using face fluxes
        U_to(b, i, j, k) = U_from(b, i, j, k) - (dt/dx)*(fx(b, fi + 1, fj, k) - fx(b, fi, fj, k))
                                              - (dt/dy)*(fy(b, fi, fj + 1, k) - fy(b, fi, fj, k));
      }
    }
  );
}

void CombineFluxDivergence(Mesh& mesh, const View4D& U_a, const View4D& U_b, const View4D& U_to,
                           const FaceFluxes& F, Real dt, Real ca, Real cb){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;
  const int nx1 = mesh.inputs().nx1_block;
  const int nx2 = mesh.inputs().nx2_block;
  const int ng  = mesh.inputs().nghost;
  const int is = ng, ie = ng + nx1 - 1;
  const int js = ng, je = ng + nx2 - 1;
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  const View4D fx = F.fx, fy = F.fy;

  Kokkos::parallel_for("CombineFluxDivergence",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      const Real dx = block_dx1(b), dy = block_dx2(b);
      const int fi = i - is, fj = j - js;
      for(size_t k = 0; k < NVAR; ++k){
        const Real ub = U_b(b, i, j, k) - (dt/dx)*(fx(b, fi + 1, fj, k) - fx(b, fi, fj, k))
                                        - (dt/dy)*(fy(b, fi, fj + 1, k) - fy(b, fi, fj, k));
        U_to(b, i, j, k) = ca*U_a(b, i, j, k) + cb*ub;
      }
    }
  );
}
