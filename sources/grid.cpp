#include "grid.hpp"
#include "hydro_config.hpp"
#include "mesh.hpp"
#include <stdexcept>

//validates that periodic boundary conditions are symmetrically paired
//across opposing domain walls (left/right and bottom/top).
void validateBC(const GridBC& bc){
  if((bc.left.type == BCType::Periodic) != (bc.right.type == BCType::Periodic))
    throw std::invalid_argument("Periodic BC must be applied to both left and right walls");
  if((bc.bottom.type == BCType::Periodic) != (bc.top.type == BCType::Periodic))
    throw std::invalid_argument("Periodic BC must be applied to both bottom and top walls");
}


//applies physical boundary conditions across all active mesh blocks that 
//border the external domain boundaries using a Kokkos parallel kernel.
//Supports Open, Closed (reflecting momentum), and Dirichlet BCs.
void ApplyPhysicalBoundaryConditions(Mesh& mesh, const GridBC& bc) {
  validateBC(bc);

  int n_active = mesh.num_active_blocks();
  if (n_active == 0) return;

  View4D data = mesh.get_data_view();
  IntView2D n_level = mesh.get_neighbor_level_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;

  //boundary conditions packed into POD for lambda capture
  GridBC bc_copy = bc;

  //physical boundary condition kernel operating across ALL active blocks
  Kokkos::parallel_for(
    "ApplyPhysicalBC",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_active, std::max(nx1, nx2) + 2*nghost}),
    KOKKOS_LAMBDA(const int b, const int t){
      //Left boundary
      if(n_level(b, (int)FaceDir::XM) == (int)NeighborLevel::PhysicalBoundary && t < (nx2 + 2*nghost)){
        int j = js - nghost + t;
        for(int g = 0; g < nghost; ++g){
          int ghost_i = is - 1 - g;
          switch(bc_copy.left.type){
            case BCType::Open:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, is, j, k);
              break;
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, is + g, j, k);
              data(b, ghost_i, j, IM1) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = bc_copy.left.Q_fixed[k];
              break;
            case BCType::Outflow:{
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, is, j, k);
              Real mn = data(b, ghost_i, j, IM1);
              if(mn > 0.0){ //inflow -> remove normal momentum and its kinetic energy
                data(b, ghost_i, j, IEN) -= 0.5*mn*mn/data(b, ghost_i, j, IDN);
                data(b, ghost_i, j, IM1) = 0.0;
              }
              break;
            }
            default: break;
          }
        }
      }

      //Right
      if(n_level(b, (int)FaceDir::XP) == (int)NeighborLevel::PhysicalBoundary && t < (nx2 + 2 * nghost)){
        int j = js - nghost + t;
        for (int g = 0; g < nghost; ++g){
          int ghost_i = ie + 1 + g;
          switch(bc_copy.right.type){
            case BCType::Open:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, ie, j, k);
              break;
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, ie - g, j, k);
              data(b, ghost_i, j, IM1) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = bc_copy.right.Q_fixed[k];
              break;
            case BCType::Outflow:{
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, ie, j, k);
              Real mn = data(b, ghost_i, j, IM1);
              if(mn < 0.0){ //inflow -> remove normal momentum and its kinetic energy
                data(b, ghost_i, j, IEN) -= 0.5*mn*mn/data(b, ghost_i, j, IDN);
                data(b, ghost_i, j, IM1) = 0.0;
              }
              break;
            }
            default: break;
          }
        }
      }

      //Bottom
      if(n_level(b, (int)FaceDir::YM) == (int)NeighborLevel::PhysicalBoundary && t < (nx1 + 2 * nghost)){
        int i = is - nghost + t;
        for(int g = 0; g < nghost; ++g){
          int ghost_j = js - 1 - g;
          switch(bc_copy.bottom.type){
            case BCType::Open:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, js, k);
              break;
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, js + g, k);
              data(b, i, ghost_j, IM2) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = bc_copy.bottom.Q_fixed[k];
              break;
            case BCType::Outflow:{
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, js, k);
              Real mn = data(b, i, ghost_j, IM2);
              if(mn > 0.0){ //inflow -> remove normal momentum and its kinetic energy
                data(b, i, ghost_j, IEN) -= 0.5*mn*mn/data(b, i, ghost_j, IDN);
                data(b, i, ghost_j, IM2) = 0.0;
              }
              break;
            }
            default: break;
          }
        }
      }

      //Top
      if(n_level(b, (int)FaceDir::YP) == (int)NeighborLevel::PhysicalBoundary && t < (nx1 + 2 * nghost)){
        int i = is - nghost + t;
        for(int g = 0; g < nghost; ++g){
          int ghost_j = je + 1 + g;
          switch(bc_copy.top.type){
            case BCType::Open:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, je, k);
              break;
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, je - g, k);
              data(b, i, ghost_j, IM2) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = bc_copy.top.Q_fixed[k];
              break;
            case BCType::Outflow:{
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, je, k);
              Real mn = data(b, i, ghost_j, IM2);
              if(mn < 0.0){ //inflow -> remove normal momentum and its kinetic energy
                data(b, i, ghost_j, IEN) -= 0.5*mn*mn/data(b, i, ghost_j, IDN);
                data(b, i, ghost_j, IM2) = 0.0;
              }
              break;
            }
            default: break;
          }
        }
      }
    }
  );
}
