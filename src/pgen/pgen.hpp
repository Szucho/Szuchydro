#pragma once

#include <functional>
#include <string>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"
#include "io/hydro_io.hpp"

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
  //operator-split source terms, applied after every hydro step (e.g. stirring, cooling)
  std::function<void(Mesh& mesh, Real dt)> source_terms;
  //extra ghost-cell treatment, applied right after the physical boundary conditions
  std::function<void(Mesh& mesh)> boundaries;
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

//list extra problems below this comment
