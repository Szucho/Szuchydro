#include "pgen/pgen.hpp"
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

/*
 Kolmogorov flow (gas_profile kolmogorov): a periodic box driven by the steady acceleration
   a = (kol_F sin(k y), 0),   k = 2 pi kol_n / Ly
 with linear drag -kol_drag u and Newtonian cooling towards kol_T0. 
 Viscosity comes from the diffusion module (viscosity_nu, viscosity_bulk)
 the laminar steady state is
   u = U sin(k y),   U = kol_F / (nu k^2 + kol_drag).

 Sources and how they are integrated:
   forcing   rho a (momentum), rho u.a (energy); inside every stage (smooth_sources)
   drag      -kol_drag rho u (momentum), -kol_drag rho |u|^2 (energy); inside every stage, second order
             with the scheme; the run stops if kol_drag dt > 0.5 (explicit limit; super-time-stepping
             is planned for strong drag)
   cooling   T - kol_T0 multiplied by exp(-dt/kol_cool_tau) once per step at constant rho and u
             (exact for any kol_cool_tau, also much shorter than dt); 0 = no cooling

 Initial state: kol_init laminar (the steady profile, needs nu or kol_drag > 0) or rest, plus an
 optional uniform flow gas_u0, gas_v0. kol_noise > 0 adds a divergence-free perturbation built from a
 stream function of random modes with wavenumbers 1..4 in each direction, scaled to an rms velocity of
 kol_noise times the sound speed; kol_seed makes it reproducible on every compiler and device
 (raw std::mt19937_64 output is fixed by the C++ standard, the conversion to doubles is done here,
  also I had to practice with GCoL and Monte Carlo simulations in order to understand the random package T^T)
*/
void ProblemKolmogorov(Mesh& mesh, const SimParams& par, ProblemHooks& hooks){
  const ParamStore& s = par.store;
  const Real gamma = par.gamma, gm1 = par.gamma - 1.0;
  const Real rho0 = s.get_real("gas_rho0"), p0 = s.get_real("gas_p0");
  const Real F = s.get_real("kol_F");
  const int n = (int)s.get_real("kol_n", 1.0);
  const Real alpha = s.get_real("kol_drag", 0.0);
  const Real tau = s.get_real("kol_cool_tau", 0.0);
  const Real T0 = s.get_real("kol_T0", p0/rho0);
  const std::string init = s.get("kol_init", "laminar");
  const Real noise = s.get_real("kol_noise", 0.0);
  const std::uint64_t seed = (std::uint64_t)s.get_real("kol_seed", 1.0);
  const Real u0 = s.get_real("gas_u0", 0.0), v0 = s.get_real("gas_v0", 0.0);
  if(alpha < 0.0 || tau < 0.0 || noise < 0.0 || T0 <= 0.0 || n < 1) //error message of doom and despair
    throw std::runtime_error("Parameter error: kol_drag, kol_cool_tau, kol_noise must be >= 0, kol_T0 > 0, kol_n >= 1");
  if(init != "laminar" && init != "rest")
    throw std::runtime_error("Parameter error: kol_init must be laminar or rest, got '" + init + "'");
  const Real lx = par.xmax - par.xmin, ly = par.ymax - par.ymin;
  const Real k = 2.0*M_PI*n/ly;
  const Real nu = par.viscosity_nu;
  if(init == "laminar" && nu*k*k + alpha <= 0.0)
    throw std::runtime_error("Parameter error: kol_init laminar needs viscosity_nu > 0 or kol_drag > 0");
  const Real U = (init == "laminar") ? F/(nu*k*k + alpha) : 0.0;
  const Real cs = std::sqrt(gamma*p0/rho0);

  //random stream-function modes (wavenumbers 1..4 in x and y), reproducible from kol_seed as per usual with random
  struct Mode{ int mx, my; Real amp, phase; };
  std::vector<Mode> modes;
  if(noise > 0.0){
    std::mt19937_64 gen(seed); //the "multi threading issue" of the generator is solved this way
                               //basically don't use the same one generator for every thread
    auto uniform01 = [&gen](){ return (Real)(gen() >> 11)*(1.0/9007199254740992.0); };
    for(int mx = -4; mx <= 4; ++mx)
      for(int my = 0; my <= 4; ++my){
        if(my == 0 && mx <= 0) continue; //each (mx, my) pair once, no mean mode
        modes.push_back({mx, my, uniform01() - 0.5, 2.0*M_PI*uniform01()});
      }
  }
  //velocity of the perturbation at (x, y): u = d(psi)/dy, v = -d(psi)/dx
  auto pert = [&](Real x, Real y, Real& up, Real& vp){
    up = 0.0;
    vp = 0.0;
    for(const Mode& m : modes){
      const Real kx = 2.0*M_PI*m.mx/lx, ky = 2.0*M_PI*m.my/ly;
      const Real c = m.amp*std::cos(kx*(x - par.xmin) + ky*(y - par.ymin) + m.phase);
      up += ky*c;
      vp -= kx*c;
    }
  };
  //rms of the raw perturbation on the base grid, to scale it to kol_noise * cs
  Real scale = 0.0;
  if(noise > 0.0){
    const int sx = 64, sy = 64;
    Real sum = 0.0;
    for(int i = 0; i < sx; ++i)
      for(int j = 0; j < sy; ++j){
        Real up, vp;
        pert(par.xmin + (i + 0.5)*lx/sx, par.ymin + (j + 0.5)*ly/sy, up, vp);
        sum += up*up + vp*vp;
      }
    scale = noise*cs/std::sqrt(sum/(sx*sy));
  }

  //applying it to the grid (sort of with the mirror view)
  View4D data = mesh.get_data_view();
  auto h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(h, data);
  const int ng = mesh.inputs().nghost;
  const int ni = mesh.inputs().nx1_block + 2*ng, nj = mesh.inputs().nx2_block + 2*ng;
  for(const auto& bp : mesh.blocks()){
    const MeshBlock& b = *bp;
    for(int i = 0; i < ni; ++i){
      const Real x = b.block_size.x1min + (i - b.is + 0.5)*b.dx1();
      for(int j = 0; j < nj; ++j){
        const Real y = b.block_size.x2min + (j - b.js + 0.5)*b.dx2();
        Real up = 0.0, vp = 0.0;
        if(noise > 0.0){ pert(x, y, up, vp); up *= scale; vp *= scale; }
        const Real u = u0 + U*std::sin(k*(y - par.ymin)) + up, v = v0 + vp;
        h(b.slot_id, i, j, IDN) = rho0;
        h(b.slot_id, i, j, IM1) = rho0*u;
        h(b.slot_id, i, j, IM2) = rho0*v;
        h(b.slot_id, i, j, IEN) = p0/gm1 + 0.5*rho0*(u*u + v*v);
      }
    }
  }
  Kokkos::deep_copy(data, h); //here we apply changes

  //forcing and drag inside every stage, from the stage state
  const Real ymin = par.ymin;
  hooks.smooth_sources = [F, k, alpha, ymin](Mesh& m, const View4D& Us, const View4D& Uo, Real dt){
    const int na = m.num_active_blocks();
    if(na == 0) return;
    const int g = m.inputs().nghost, nx1 = m.inputs().nx1_block, nx2 = m.inputs().nx2_block;
    View1D bdx2 = m.get_block_dx2_view();
    View1D bx2min("kol_block_x2min", na); //lower y edge of every active block (valid after regrids)
    auto hx = Kokkos::create_mirror_view(bx2min);
    const auto& blocks = m.blocks();
    for(int b = 0; b < na; ++b) hx(b) = blocks[b]->block_size.x2min;
    Kokkos::deep_copy(bx2min, hx);
    Kokkos::parallel_for("KolmogorovSources",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, g, g}, {na, g + nx1, g + nx2}),
      KOKKOS_LAMBDA(const int b, const int i, const int j){
        const Real y = bx2min(b) + (j - g + 0.5)*bdx2(b);
        const Real a = F*Kokkos::sin(k*(y - ymin));
        const Real rho = Us(b, i, j, IDN), mx = Us(b, i, j, IM1), my = Us(b, i, j, IM2);
        Uo(b, i, j, IM1) += dt*(rho*a - alpha*mx);
        Uo(b, i, j, IM2) += dt*(-alpha*my);
        Uo(b, i, j, IEN) += dt*(mx*a - alpha*(mx*mx + my*my)/rho);
      });
  };

  //the drag is explicit: stop cleanly instead of going unstable
  //TODO: up for debate on this but this one might need a revisit
  hooks.begin_step = [alpha](Mesh&, Real dt){
    if(alpha*dt > 0.5)
      throw std::runtime_error("kol_drag dt = " + std::to_string(alpha*dt) + " > 0.5: the drag is too strong for "
                               "its explicit integration; lower kol_drag or cfl (super-time-stepping)");
  };

  //Newtonian cooling, exact factor once per step
  if(tau > 0.0)
    hooks.source_terms = [tau, T0, gm1](Mesh& m, Real dt){
      const int na = m.num_active_blocks();
      if(na == 0) return;
      const int g = m.inputs().nghost, nx1 = m.inputs().nx1_block, nx2 = m.inputs().nx2_block;
      View4D U = m.get_data_view();
      const Real f = Kokkos::exp(-dt/tau);
      Kokkos::parallel_for("KolmogorovCooling",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, g, g}, {na, g + nx1, g + nx2}),
        KOKKOS_LAMBDA(const int b, const int i, const int j){
          const Real rho = U(b, i, j, IDN), mx = U(b, i, j, IM1), my = U(b, i, j, IM2);
          const Real ke = 0.5*(mx*mx + my*my)/rho;
          const Real T = gm1*(U(b, i, j, IEN) - ke)/rho;
          U(b, i, j, IEN) = ke + rho*(T0 + (T - T0)*f)/gm1;
        });
    };
}
