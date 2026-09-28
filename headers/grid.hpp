#pragma once

#include <stdexcept>
#include "hydro_config.hpp"


/*
 Grid BC application for 2D Grids
 Bertalan Szuchovszky 02.03.2026

 Cell: cell point struct, at each cell point we have a Q state vector, Cell[0] = rho, Cell[1] = rho u ...
 Grid: the whole grid, every element of the grid is a cell Grid(i,j)[0] = rho, Grid(i,j)[1] = rho u ...
 BCType: Boundary condition type - just names, the solver just needs the names
 BoundaryCondition: sets the BCType, for Dirichlet we need Cell Q_fixed constant state vector
 GridBC: BC type at each grid wall

 The other 2 functions are defined in grid_setup.cpp
 

 MODIFIED 12.05.2026
 -Grid is i*j*k 1D Vector, cell is a temporary value holder.
 -Added some helper functions into Grid so that the new cell can be accessed.

 Updated: September 2026
 -added KOKKOS to the mix
*/


//4D View storing all mesh blocks: (block_slot, i, j, var)
using MeshDataView = Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>;

void validateBC(const GridBC& bc);
