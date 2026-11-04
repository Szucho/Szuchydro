#pragma once

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <map>
#include <iomanip>
#include <cstdint>
#include <vector>
#include "hydro_config.hpp"
#include "grid.hpp"
#include "mesh.hpp"
#include "disk_physics.hpp"

/*
 Input / Output header for Kokkos AMR Hydro Simulation
 Bertalan Szuchovszky 20.04.2026
 
 Modified on 21.09.2026:
  - Adapted to parse full params.txt configuration files.
  - Added support for exporting binary grid structures and simulation frames directly from flat Kokkos device views

*/

//structure holding all simulation, grid, physics, and I/O parameters parsed from config files
struct SimParams {
  //grid
  double Nx, Ny, Nt;
  double t0, tf;
  double xmin, xmax, ymin, ymax;
  //refinement
  int nx1_block = 0, nx2_block = 0;
  int nghost = 2;
  std::vector<RefinementRegion> refine_regions;
  //physics
  double gamma;
  //boundary conditions
  std::string bc_left, bc_right, bc_top, bc_bottom;
  //gas initial condition
  std::string gas_profile;
  double gas_rho0, gas_p0, gas_u0, gas_v0; 
  //output
  std::string outdir;
  int n_frames = 100; //number of snapshots written to grid.bin
  //run control
  bool use_amr = true; //false -> static mesh (SMR only)
  double cfl = 0.4;
  //disk r--z problem:
  DiskParams disk;
};

//parameter and setup functions
SimParams readParams(const std::string& filename);
BCType parseBC(const std::string& s);
void init_cond(Mesh& mesh, const SimParams& par);
DiskModel BuildDiskModelFromParams(const SimParams& par);
void setupOutputDir(const std::string& dir);
void writeMetadata(const std::string& outdir, const SimParams& par);
std::ofstream openGridFile(const std::string& outdir, Mesh& mesh);
void writeFrame(std::ofstream& f, Mesh& mesh, double t);
