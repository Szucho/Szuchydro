#pragma once

#include <functional>
#include <string>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "io/hydro_io.hpp"
#include "io/history.hpp"
#include "solver/time_integrator.hpp"
#include "solver/potential.hpp"

/*
 Problem generators: one file per problem in src/pgen/, chosen with "gas_profile <name>".

 A problem generator
   - reads its own parameters from par.store (e.g. "gas_rho0", "sedov_E"),
   - fills the initial state on all blocks (ghost cells included),
   - can attach problem-specific extras to the time loop through ProblemHooks,
     so main.cpp never needs to know about any particular problem.

 NEW PROBLEM TYPES
 ADDING A PROBLEM: write src/pgen/<name>.cpp with a function like the ones below,
 declare it here and add one line to the table in pgen.cpp. CMake picks up the file by itself.
*/


//first of all we need a problem hook to main
struct ProblemHooks{
  //external potential phi(x, y) (gravity or any conservative force field).
  //If set, the solver tabulates it on the mesh and applies -rho grad(phi) / -rho v.grad(phi)
  //"well_balanced 1" keeps this problem's hydrostatic equilibria exactly at rest,
  //and "Hydrostatic" walls extend them
  PotentialFn potential;

  //once per step BEFORE the integrator (e.g. advancing random forcing)
  //with a multi-stage integrator, everyting will still happen once per step
  std::function<void(Mesh& mesh, Real dt)> begin_step;
  //smooth/well behaved source terms INSIDE the update,
  //from the same stage state as fluxes
  //(e.g. gravity, forcing, ...): add dt*S(U_stage) to U_out
  //Evaluated in every stage of the integrator
  SourceFn smooth_sources;
  //operator-split source terms, applied after every hydro step (e.g. stirring, cooling)
  std::function<void(Mesh& mesh, Real dt)> source_terms;
  //extra ghost-cell treatment, applied right after the physical boundary conditions
  std::function<void(Mesh& mesh)> boundaries;
  //"amr_criterion hook": the problem decides refinement from a block's region,
  //level and the time return > 0 refine, < 0 derefine, 0 keep
  std::function<int(Real x1min, Real x1max, Real x2min, Real x2max, int level, Real t)> refine;
  //history output (hist_interval): the problem's own volume-integrated rates 
  //(forcing power, drag loss, cooling, ...) 
  //with their role in the kinetic-energy budget; evaluated every step
  HistoryHook history;
  //diffusion_integrator rkl2: a constant linear drag coefficient alpha (-alpha m, -alpha |m|^2/rho) that the
  //RKL2 substeps integrate together with the diffusion (0 = none), with explicit it is not used
  Real linear_drag = 0.0;
};

//the initial condition has to be set up
using ProblemInit = void (*)(Mesh& mesh, const SimParams& par, ProblemHooks& hooks);

//pass things for the initial condition
struct ProblemEntry{
  const char* name;        //value of gas_profile
  ProblemInit init;
  const char* description; //one line, shown in error messages
};

//the problem with this name
//unknown name -> "Parameter error" listing all available problems
//add unique usr defined problem as specified in the first multiline comment
//important to name said problem otherwise this will throw error
const ProblemEntry& FindProblem(const std::string& name);

//the problems (one file each)
//these are mostly for testing
void ProblemUniform(Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //uniform.cpp
void ProblemSodX   (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //sod.cpp
void ProblemSedov  (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //sedov.cpp
void ProblemDiskRz (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //disk_rz.cpp
void ProblemTurb   (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //turb.cpp
void ProblemAtmosphere (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //atmosphere.cpp
void ProblemAdvect (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //advect.cpp
void ProblemVortex (Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //vortex.cpp
void ProblemPointmass(Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //pointmass.cpp
void ProblemDiffusionTest(Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //diffusion_test.cpp
void ProblemKolmogorov(Mesh& mesh, const SimParams& par, ProblemHooks& hooks); //kolmogorov.cpp


//list extra problems below this comment
