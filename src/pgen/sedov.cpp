#include "pgen/pgen.hpp"
#include <stdexcept>

//Sedov-Taylor blast: uniform medium (gas_rho0, gas_p0) plus energy sedov_E (default 1),
//added on top of the ambient gas inside radius sedov_r (default 0 = 3 base cells)
void ProblemSedov(Mesh& mesh, const SimParams& par, ProblemHooks& /*hooks*/){
  const Real gamma = par.gamma;
  const Real rho0 = par.store.get_real("gas_rho0");
  const Real p0   = par.store.get_real("gas_p0");
  const Real sedov_E = par.store.get_real("sedov_E", 1.0);
  const Real sedov_r = par.store.get_real("sedov_r", 0.0);
  View4D data = mesh.get_data_view();
  const int nghost = mesh.inputs().nghost;

  //deposit exactly sedov_E: sum the area of the cells inside r_dep on the host
  //(a plain pi*r^2 would be off by the staircase of square cells)
  Real dx_base = (par.xmax - par.xmin)/par.Nx;
  Real sedov_r_dep = (sedov_r > 0.0) ? sedov_r : 3.0*dx_base;
  Real xc = 0.5*(par.xmin + par.xmax), yc = 0.5*(par.ymin + par.ymax);
  Real area = 0.0;
  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    for(int i=0; i<b.block_size.nx1; ++i){
      for(int j=0; j<b.block_size.nx2; ++j){
        Real x = b.block_size.x1min + (i + 0.5)*b.dx1() - xc;
        Real y = b.block_size.x2min + (j + 0.5)*b.dx2() - yc;
        if(x*x + y*y < sedov_r_dep*sedov_r_dep) area += b.dx1()*b.dx2();
      }
    }
  }
  if(area<=0.0) throw std::runtime_error("[SEDOV]: deposit radius smaller than a cell");
  Real sedov_p_in = (gamma - 1.0)*sedov_E/area;

  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int b_slot = b.slot_id;
    int total_i = b.block_size.nx1 + 2*nghost;
    int total_j = b.block_size.nx2 + 2*nghost;
    int is = b.is, js = b.js;
    Real x1min = b.block_size.x1min, x2min = b.block_size.x2min;
    Real dx1 = b.dx1(), dx2 = b.dx2();
    Real r_dep = sedov_r_dep, p_in = sedov_p_in;
    Kokkos::parallel_for("InitSedov", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
      KOKKOS_LAMBDA(int i, int j){
        Real x = x1min + (i - is + 0.5)*dx1 - xc;
        Real y = x2min + (j - js + 0.5)*dx2 - yc;
        Real p = (x*x + y*y < r_dep*r_dep) ? p0 + p_in : p0; //ambient + blast energy
        data(b_slot, i, j, IDN) = rho0;
        data(b_slot, i, j, IM1) = 0.0;
        data(b_slot, i, j, IM2) = 0.0;
        data(b_slot, i, j, IEN) = p/(gamma - 1.0);
      }
    );
  }
}
