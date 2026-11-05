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
  int Nx=0, Ny=0, Nt=0; //root grid cells, initial number of time steps (first dt guess)
  Real t0, tf;
  Real xmin, xmax, ymin, ymax;
  //refinement
  int nx1_block = 0, nx2_block = 0;
  int nghost = 2;
  std::vector<RefinementRegion> refine_regions;
  //physics
  Real gamma;
  //boundary conditions
  std::string bc_left, bc_right, bc_top, bc_bottom;
  //gas initial condition
  std::string gas_profile;
  Real gas_rho0, gas_p0, gas_u0, gas_v0; 
  //output
  std::string outdir;
  int n_frames = 100; //number of snapshots written to grid.bin
  //run control
  bool use_amr = true; //false -> static mesh (SMR only)
  int max_blocks = 1024; //capacity of block arrays, memory is allocated for all of them
  int amr_max_level = 3; //AMR never refines beyond this level
  int amr_interval = 5;  //check refinement level every amr interval steps
  Real amr_refine_tol = 0.05; //refine a block if the relative density jump exeeds this tolerance value
  Real amr_derefine_tol = 0.01; //derefine if it stays below this everywhere
  Real cfl = 0.4;
  //disk r--z problem:
  DiskParams disk;

  //Sedov problem
  Real sedov_E = 1.0; 
  Real sedov_r = 0.0;
};

//parameter and setup functions
SimParams readParams(const std::string& filename);
BCType parseBC(const std::string& s);
void init_cond(Mesh& mesh, const SimParams& par);
DiskModel BuildDiskModelFromParams(const SimParams& par);
void setupOutputDir(const std::string& dir);
void writeMetadata(const std::string& outdir, const SimParams& par);
std::ofstream openGridFile(const std::string& outdir, Mesh& mesh);
void writeFrame(std::ofstream& f, Mesh& mesh, Real t);
