#include "solver/diffusion.hpp"
#include <algorithm>
#include <limits>

namespace {
  //velocity component m (IM1 or IM2) of a cell from conserved vars
  KOKKOS_INLINE_FUNCTION
  Real vel(const View4D& U, int b, int i, int j, int m){ return U(b, i, j, m)/U(b, i, j, IDN); }

  //temperature T = p/rho of a cell from conserved vars
  KOKKOS_INLINE_FUNCTION
  Real temperature(const View4D& U, int b, int i, int j, Real gm1){
    const Real rho = U(b, i, j, IDN), mx = U(b, i, j, IM1), my = U(b, i, j, IM2);
    return gm1*(U(b, i, j, IEN) - 0.5*(mx*mx + my*my)/rho)/rho;
  }

  //ddy, ddx are tangential derivatives
  //central differences, one-sided in a block's first and last row (column)
  //d(velocity m)/dy at cell (i, j): central, one-sided in the block's first/last row (no corner cells)
  KOKKOS_INLINE_FUNCTION
  Real ddy(const View4D& U, int b, int i, int j, int m, int js, int je, Real dy){
    const int jm = (j > js) ? j - 1 : j, jp = (j < je) ? j + 1 : j;
    return (vel(U, b, i, jp, m) - vel(U, b, i, jm, m))/((jp - jm)*dy);
  }

  //d(velocity m)/dx at cell (i, j): central, one-sided in the block's first/last column
  KOKKOS_INLINE_FUNCTION
  Real ddx(const View4D& U, int b, int i, int j, int m, int is, int ie, Real dx){
    const int im = (i > is) ? i - 1 : i, ip = (i < ie) ? i + 1 : i;
    return (vel(U, b, ip, j, m) - vel(U, b, im, j, m))/((ip - im)*dx);
  }
}

//marks the faces that are Closed or Hydrostatic physical walls
//There no diffusive flux is added: free-slip and insulating
WallFlags ReflectingWallFlags(Mesh& mesh, const GridBC& bc){
  const int n_active = mesh.num_active_blocks();
  auto refl = [](BCType t){ return t == BCType::Closed || t == BCType::Hydrostatic; };
  const bool wall[4] = {refl(bc.left.type), refl(bc.right.type), refl(bc.bottom.type), refl(bc.top.type)};
  if(n_active == 0 || !(wall[0] || wall[1] || wall[2] || wall[3])) return WallFlags();
  WallFlags w("reflecting_walls", n_active);
  auto h = Kokkos::create_mirror_view(w);
  const auto& blocks = mesh.blocks();
  for(int b = 0; b < n_active; ++b)
    for(int f = 0; f < 4; ++f) h(b, f) = (blocks[b]->is_physical_boundary[f] && wall[f]) ? 1 : 0;
  Kokkos::deep_copy(w, h);
  return w;
}

//has two kernels (x- and y-faces), each computing the stress and heat flux at the face and adds
//-tau*n, -(tau*u)*n + q*n to face arrays and flux registers at block edges for refluxing
void AddDiffusiveFluxes(Mesh& mesh, const View4D& U, Real gamma, const DiffusionCoeffs& d,
                        const FaceFluxes& F, const WallFlags& walls){
  const int n_active = mesh.num_active_blocks();
  if(n_active == 0 || !d.active()) return;
  const int ng = mesh.inputs().nghost, nx1 = mesh.inputs().nx1_block, nx2 = mesh.inputs().nx2_block;
  const int is = ng, ie = ng + nx1 - 1, js = ng, je = ng + nx2 - 1;
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  View4D reg = mesh.get_flux_view();
  const View4D fx = F.fx, fy = F.fy;
  const Real nu = d.nu, nub = d.nu_bulk, chi = d.chi;
  const Real gm1 = gamma - 1.0, cp = gamma/gm1;
  const bool nw = walls.extent(0) > 0;
  const WallFlags wl = walls;

  //x-faces: face f between cells (i - 1, j) and (i, j), i = is + f
  Kokkos::parallel_for("DiffusiveFluxesX",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, nx1 + 1, nx2}),
    KOKKOS_LAMBDA(const int b, const int f, const int jj){
      if(nw && ((f == 0 && wl(b, 0)) || (f == nx1 && wl(b, 1)))) return; //free-slip, insulating wall
      const int i = is + f, j = js + jj;
      const Real dx = block_dx1(b), dy = block_dx2(b);
      const Real rf = 0.5*(U(b, i - 1, j, IDN) + U(b, i, j, IDN));
      const Real ul = vel(U, b, i - 1, j, IM1), ur = vel(U, b, i, j, IM1);
      const Real vl = vel(U, b, i - 1, j, IM2), vr = vel(U, b, i, j, IM2);
      const Real dudx = (ur - ul)/dx, dvdx = (vr - vl)/dx;
      const Real dudy = 0.5*(ddy(U, b, i - 1, j, IM1, js, je, dy) + ddy(U, b, i, j, IM1, js, je, dy));
      const Real dvdy = 0.5*(ddy(U, b, i - 1, j, IM2, js, je, dy) + ddy(U, b, i, j, IM2, js, je, dy));
      const Real div = dudx + dvdy;
      const Real mu = rf*nu, mub = rf*nub;
      const Real txx = 2.0*mu*dudx - (2.0/3.0)*mu*div + mub*div;
      const Real txy = mu*(dudy + dvdx);
      const Real qx = -rf*cp*chi*(temperature(U, b, i, j, gm1) - temperature(U, b, i - 1, j, gm1))/dx;
      const Real dF[4] = {0.0, -txx, -txy, -(txx*0.5*(ul + ur) + txy*0.5*(vl + vr)) + qx};
      for(int k = 1; k < 4; ++k){
        fx(b, f, jj, k) += dF[k];
        if(f == 0) reg(b, (int)FaceDir::XM, jj, k) += dF[k];
        if(f == nx1) reg(b, (int)FaceDir::XP, jj, k) += dF[k];
      }
    });

  //y-faces: face f between cells (i, j - 1) and (i, j), j = js + f
  Kokkos::parallel_for("DiffusiveFluxesY",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, nx1, nx2 + 1}),
    KOKKOS_LAMBDA(const int b, const int ii, const int f){
      if(nw && ((f == 0 && wl(b, 2)) || (f == nx2 && wl(b, 3)))) return; //free-slip, insulating wall
      const int i = is + ii, j = js + f;
      const Real dx = block_dx1(b), dy = block_dx2(b);
      const Real rf = 0.5*(U(b, i, j - 1, IDN) + U(b, i, j, IDN));
      const Real ul = vel(U, b, i, j - 1, IM1), ur = vel(U, b, i, j, IM1);
      const Real vl = vel(U, b, i, j - 1, IM2), vr = vel(U, b, i, j, IM2);
      const Real dudy = (ur - ul)/dy, dvdy = (vr - vl)/dy;
      const Real dudx = 0.5*(ddx(U, b, i, j - 1, IM1, is, ie, dx) + ddx(U, b, i, j, IM1, is, ie, dx));
      const Real dvdx = 0.5*(ddx(U, b, i, j - 1, IM2, is, ie, dx) + ddx(U, b, i, j, IM2, is, ie, dx));
      const Real div = dudx + dvdy;
      const Real mu = rf*nu, mub = rf*nub;
      const Real tyy = 2.0*mu*dvdy - (2.0/3.0)*mu*div + mub*div;
      const Real txy = mu*(dudy + dvdx);
      const Real qy = -rf*cp*chi*(temperature(U, b, i, j, gm1) - temperature(U, b, i, j - 1, gm1))/dy;
      const Real dF[4] = {0.0, -txy, -tyy, -(txy*0.5*(ul + ur) + tyy*0.5*(vl + vr)) + qy};
      for(int k = 1; k < 4; ++k){
        fy(b, ii, f, k) += dF[k];
        if(f == 0) reg(b, (int)FaceDir::YM, ii, k) += dF[k];
        if(f == nx2) reg(b, (int)FaceDir::YP, ii, k) += dF[k];
      }
    });
}


//cfl condition for diffusion for the numerical stability of the explicit solver
Real DiffusiveTimeStep(Mesh& mesh, Real gamma, const DiffusionCoeffs& d, Real cfl_diff){
  const Real D = std::max((4.0/3.0)*d.nu + d.nu_bulk, gamma*d.chi);
  if(D <= 0.0) return std::numeric_limits<Real>::max();
  Real dt = std::numeric_limits<Real>::max();
  for(const auto& bp : mesh.blocks()){
    const Real dx = bp->dx1(), dy = bp->dx2();
    dt = std::min(dt, cfl_diff*0.5/(D*(1.0/(dx*dx) + 1.0/(dy*dy))));
  }
  return dt;
}
