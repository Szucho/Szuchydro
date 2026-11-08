#include "pgen/pgen.hpp"
#include <stdexcept>

//all available problems 
//an explicit table (rather than self-registering files) is easy to read,
//and a static library cannot silently drop a problem whose file nothing references.
static const ProblemEntry problems[] = {
  {"uniform", ProblemUniform, "uniform state gas_rho0, gas_p0, gas_u0, gas_v0"},
  {"Sod_x",   ProblemSodX,    "Sod shock tube along x (left: gas_rho0, gas_p0; right: 1/8, 1/10 of them)"},
  {"sedov",   ProblemSedov,   "Sedov-Taylor blast: energy sedov_E inside radius sedov_r"},
  {"disk_Rz", ProblemDiskRz,  "stirred protoplanetary disk with winds in the R-z plane"},
  {"turb",    ProblemTurb,    "driven turbulence in a periodic box (few-mode OU forcing, turb_* keys)"},
  {"atmosphere", ProblemAtmosphere, "stratified layer in hydro equilibrium in external potential (atm_*keys)"},
};

//here we find the problem we wish to solve
const ProblemEntry& FindProblem(const std::string& name){
  for(const auto& p : problems)
    if(name == p.name) return p;
  std::string list;
  for(const auto& p : problems) list += std::string("\n  ") + p.name + ": " + p.description;
  throw std::runtime_error("Parameter error: unknown gas_profile '" + name + "'. Available:" + list);
}
