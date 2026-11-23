#pragma once

#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "core/hydro_config.hpp"
#include "mesh/mesh.hpp"

/*
 History output: <outdir>/history.hst, a plain-text table with a '#' header naming the
 columns (np.loadtxt reads it directly).

 Every step, at the top of the step (ghost cells filled, the state writeFrame would save), the
 recorder measures volume integrals and rates; the rates are integrated in time with the trapezoidal
 rule EVERY step, so the cumulative columns do not depend on how often lines are written
 (hist_interval, a line every N steps, plus the final state).

 Columns:
   t dt mass mx my Ekin Eint Etot enstrophy   state integrals (enstrophy = 1/2 int omega^2)
   D_visc W_p                                  viscous dissipation int tau:grad u, pressure work int p div u
   <problem rates>                             from the problem's history hook (e.g. P_force D_drag Q_cool)
   int_<rate> ...                              cumulative time integrals of all rates
   eps_num                                     cumulative numerical dissipation of kinetic energy:
                                               int (sum of kinetic rates - D_visc + W_p) dt - (Ekin - Ekin0)
 Derivatives are central differences of the cell velocities (O(dx^2) from the solver's own discrete
 operators), and the time integrals are second order, so eps_num is accurate to ~(dx^2, dt^2).
*/
struct HistoryTerm{
  std::string name;
  Real rate;    //volume-integrated rate
  int kinetic;  //+1 gains kinetic energy (forcing), -1 removes it (drag), 0 not in the kinetic budget (cooling)
};

using HistoryHook = std::function<std::vector<HistoryTerm>(Mesh& mesh, Real gamma)>;

class HistoryRecorder{
  public:
    //interval 0: no history at all (every call returns immediately)
    HistoryRecorder(const std::string& outdir, int interval, Real gamma,
                    Real nu, Real nu_bulk, const HistoryHook& hook);

    //measure, integrate the rates from the previous sample, and write a line if asked
    void record(Mesh& mesh, Real t, bool write_line);
    bool active() const { return interval_ > 0; }
    int interval() const { return interval_; }

  private:
    int interval_;
    Real gamma_, nu_, nub_;
    HistoryHook hook_;
    std::ofstream file_;
    bool first_ = true;
    Real t_prev_ = 0.0, ekin0_ = 0.0;
    std::vector<std::string> rate_names_;
    std::vector<int> kinetic_;
    std::vector<Real> rates_prev_, integrals_;
};
