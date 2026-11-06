#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cmath>

/*
 * This guy handles everything for Kokkos
 * Defines core aliases, dimension parameters, and execution/memory spaces 
 * optimized for multi-backend (CPU/GPU) execution via Kokkos
 * 
*/

using Real = double; //allows for precision switching

static constexpr std::size_t NVAR = 4; // [rho, rho*u, rho*v, E]

//execution and memory space abstractions mapped to the default Kokkos backend
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace    = ExecutionSpace::memory_space;

//multi-dimensional Kokkos View aliases for block-structured grid management
using View1D = Kokkos::View<Real*, Kokkos::LayoutRight, MemorySpace>;
using View3D = Kokkos::View<Real***, Kokkos::LayoutRight, MemorySpace>;
using View4D = Kokkos::View<Real****, Kokkos::LayoutRight, MemorySpace>;

using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, MemorySpace>;

//host mirror views for data transfer and CPU-side inspection I/O
using HostView1D    = typename View1D::host_mirror_type;
using HostView3D    = typename View3D::host_mirror_type;
using HostView4D    = typename View4D::host_mirror_type;
using HostIntView2D = typename IntView2D::host_mirror_type;

//fixed-size 4-element vector structure designed with Kokkos inline decorators
//to enable zero-allocation, high-performance execution on both host and device

struct Vec4 {
  Real data[NVAR];
  KOKKOS_INLINE_FUNCTION Real operator[](size_t k) const { return data[k]; }
  KOKKOS_INLINE_FUNCTION Real& operator[](size_t k) { return data[k]; }
};


//supported boundary condition types
enum class BCType{Open, Closed, Periodic, Dirichlet, Outflow};

//configuration container for an individual boundary wall
struct BoundaryCondition{
  BCType type;
  Vec4 Q_fixed{}; //Dirichlet boundary condition value
};

//container grouping boundary conditions for all four walls of a grid block
struct GridBC{
  BoundaryCondition left, right, top, bottom;
};
