#include <cmath>
#include <exception>
#include <iomanip>
#include <ios>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <limits>
#include <Kokkos_Core.hpp>
#include <memory>

#include "core/hydro_config.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hllc.hpp"
#include "hydro/slope_limiters.hpp"
#include "hydro/reconstruction.hpp"
#include "io/hydro_io.hpp"
#include "pgen/pgen.hpp"

//CFL time step calculation (GPU/CPU parallel reduction)
Real compute_cfl_dt(Mesh& mesh, Real gamma, Real cfl_max){
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 1.0;

  View4D data = mesh.get_data_view();
  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;

  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();

  Real max_wave_speed = 0.0;

  Kokkos::parallel_reduce(
    "ComputeCFL",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& local_max){
      Real dx1 = block_dx1(b);
      Real dx2 = block_dx2(b);

      Vec4 q_cell;
      for(size_t k = 0; k < NVAR; ++k){
        q_cell[k] = data(b, i, j, k);
      }

      Real rho = q_cell[IDN];
      Real u = q_cell[IM1]/rho;
      Real v = q_cell[IM2]/rho;
      Real p = (gamma - 1.0)*(q_cell[IEN] - rho*0.5*(u*u + v*v));
      Real cs = Kokkos::sqrt(gamma*p/rho);

      Real speed = (Kokkos::abs(u) + cs)/dx1 + (Kokkos::abs(v) + cs)/dx2;
      if(speed > local_max) local_max = speed;
    },
    Kokkos::Max<Real>(max_wave_speed)
  );

  return cfl_max/max_wave_speed;
}

//explicit 2D Euler Finite Volume timestep
void step_hydro_solver(Mesh& mesh, Real dt, Real gamma, const ReconOptions& recon){
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;

  View4D q_old = mesh.get_data_view();
  View4D q_new = mesh.get_data_next_view();
  View4D flux_face = mesh.get_flux_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;

  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();

  const ReconOptions opt = recon; //plain struct copied into kernel
  
  Kokkos::parallel_for(
    "StepHydroSolver",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){

      Real dx = block_dx1(b);
      Real dy = block_dx2(b);

      Vec4 qi, qim1, qim2, qip1, qip2;
      Vec4 qjm1, qjm2, qjp1, qjp2;

      for (size_t k = 0; k < NVAR; ++k) {
        qi[k]   = q_old(b, i, j, k);
        qim1[k] = q_old(b, i - 1, j, k);
        qim2[k] = q_old(b, i - 2, j, k);
        qip1[k] = q_old(b, i + 1, j, k);
        qip2[k] = q_old(b, i + 2, j, k);

        qjm1[k] = q_old(b, i, j - 1, k);
        qjm2[k] = q_old(b, i, j - 2, k);
        qjp1[k] = q_old(b, i, j + 1, k);
        qjp2[k] = q_old(b, i, j + 2, k);
      }

      //face states
      Vec4 ql_imh, qr_imh, ql_iph, qr_iph;
      Vec4 ql_jmh, qr_jmh, ql_jph, qr_jph;

      reconstruct_faces(qim2, qim1, qi, qip1, qip2, dx, gamma, opt, ql_imh, qr_imh, ql_iph, qr_iph);
      reconstruct_faces(qjm2, qjm1, qi, qjp1, qjp2, dy, gamma, opt, ql_jmh, qr_jmh, ql_jph, qr_jph);

      Vec4 fx_imh = FluxhllcX(ql_imh, qr_imh, gamma);
      Vec4 fx_iph = FluxhllcX(ql_iph, qr_iph, gamma);
      Vec4 fy_jmh = FluxhllcY(ql_jmh, qr_jmh, gamma);
      Vec4 fy_jph = FluxhllcY(ql_jph, qr_jph, gamma);

      if(i == is){
        for (size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::XM, j - js, k) = fx_imh[k];
      }
      if(i == ie){
        for (size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::XP, j - js, k) = fx_iph[k];
      }
      if(j == js){
        for (size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::YM, i - is, k) = fy_jmh[k];
      }
      if(j == je){
        for (size_t k = 0; k < NVAR; ++k) flux_face(b, (int)FaceDir::YP, i - is, k) = fy_jph[k];
      }

      for(size_t k = 0; k < NVAR; ++k){
        q_new(b, i, j, k) = qi[k] - (dt/dx)*(fx_iph[k] - fx_imh[k]) 
                                  - (dt/dy)*(fy_jph[k] - fy_jmh[k]);
      }
    }
  );

  mesh.swap_data_views();
}


//for conservation check: volume integral of conserved variable k (0 = mass, 3 = energy)
double compute_total(Mesh& mesh, size_t k) {
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 0.0;

  View4D data = mesh.get_data_view();
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + mesh.inputs().nx1_block - 1;
  int js = nghost, je = nghost + mesh.inputs().nx2_block - 1;

  double total = 0.0;
  //sum over active cell indices [is, ie] x [js, je]
  Kokkos::parallel_reduce("ComputeTotal",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, double& local_sum){
      local_sum += data(b, i, j, k) * block_dx1(b) * block_dx2(b);
    },
    total
  );
  return total;
}

//net outflow rate of conserved variable k through the physical domain walls,
//integrated from the face fluxes the solver actually used in the last step
double compute_boundary_outflow(Mesh& mesh, size_t k) {
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 0.0;

  View4D flux = mesh.get_flux_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  View1D block_dx1 = mesh.get_block_dx1_view();
  View1D block_dx2 = mesh.get_block_dx2_view();
  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;

  double outflow = 0.0;
  Kokkos::parallel_reduce("ComputeBoundaryOutflow",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, Kokkos::max(nx1, nx2)}),
    KOKKOS_LAMBDA(const int b, const int fi, const int t, double& local_sum){
      if(n_level(b, fi) != static_cast<int>(NeighborLevel::PhysicalBoundary)) return;
      bool xface = (fi == (int)FaceDir::XM || fi == (int)FaceDir::XP);
      if(t >= (xface ? nx2 : nx1)) return;
      Real face_len = xface ? block_dx2(b) : block_dx1(b);
      Real sign = (fi == (int)FaceDir::XP || fi == (int)FaceDir::YP) ? +1.0 : -1.0; //outward normal
      local_sum += sign * flux(b, fi, t, k) * face_len;
    },
    outflow
  );
  return outflow;
}

//relative central density gradient max(|d rho/dx|, |d rho/dy|)/rho at cell (i, j), used by the AMR criteria
KOKKOS_INLINE_FUNCTION
Real rel_density_gradient(const View4D& data, int b, int i, int j){
  Real rho_ij = data(b, i, j, IDN);
  Real grad_x = Kokkos::abs(data(b, i + 1, j, IDN) - data(b, i - 1, j, IDN))/(2.0*rho_ij + 1e-12);
  Real grad_y = Kokkos::abs(data(b, i, j + 1, IDN) - data(b, i, j - 1, IDN))/(2.0*rho_ij + 1e-12);
  return Kokkos::max(grad_x, grad_y);
}


int main(int argc, char* argv[]){
  int exit_code = 0; //not zero if run fails so scripts and CI can tell
  Kokkos::initialize(argc, argv); //GPU accel
  try{ //setup errors like bad params, invalid mesh end runs cleanly with exit code 1
    std::string param_file = (argc > 1) ? argv[1] : "params.txt"; //look for input file
    //reading and processing simulation parameters
    SimParams par = readParams(param_file);
    //unknown name: fails here, before any setup
    const ProblemEntry& problem = FindProblem(par.gas_profile); 

    MeshInputs inputs;
    inputs.domain = RegionSize{par.xmin, par.xmax, par.ymin, par.ymax, par.Nx, par.Ny};
    inputs.nx1_block = par.nx1_block;
    inputs.nx2_block = par.nx2_block;
    inputs.nghost = par.nghost;
    inputs.max_blocks = par.max_blocks;
    inputs.refine_regions = par.refine_regions;

    GridBC bc;
    bc.left.type   = parseBC(par.bc_left);
    bc.right.type  = parseBC(par.bc_right);
    bc.bottom.type = parseBC(par.bc_bottom);
    bc.top.type    = parseBC(par.bc_top);
    validateBC(bc);
    setDirichletStates(bc, par);

    inputs.periodic_x1 = (bc.left.type == BCType::Periodic);
    inputs.periodic_x2 = (bc.bottom.type == BCType::Periodic);

    Mesh mesh(inputs);
    mesh.BuildTree();
    mesh.CreateMeshBlocks();
    mesh.SetupNeighbors();

    setupOutputDir(par.outdir);
    writeMetadata(par.outdir, par);
    std::ofstream gridfile = openGridFile(par.outdir, mesh);

    std::cout << "Active leaf blocks created: " << mesh.num_active_blocks() << "\n";

    //initial state + problem-specific extras for the time loop (src/pgen/)
    ProblemHooks hooks;
    problem.init(mesh, par, hooks);

    //keys in the parameter file that nothing has read: almost always a typo
    for(const auto& key : par.store.unused())
      std::cerr << "Warning: parameter '" << key << "' is not used by this setup (typo?)\n";


    auto apply_bcs = [&](){
      ApplyPhysicalBoundaryConditions(mesh, bc);
      if(hooks.boundaries) hooks.boundaries(mesh);
    };

    //capturing initial density profile
    mesh.CaptureInitialState();
    double initial_mass = compute_total(mesh, 0);
    double initial_energy = compute_total(mesh, 3);

    Real refine_grad_threshold = par.amr_refine_tol;     //refine if local density gradient is high
    Real derefine_grad_threshold = par.amr_derefine_tol; //derefine if local gradients are quiet

    BlockCriterion density_gradient_refine = [refine_grad_threshold](const MeshBlock& block,
                                                                          const View4D& data,
                                                                          const View3D& rho_init){
      int b = block.slot_id;
      int is = block.is, ie = block.ie;
      int js = block.js, je = block.je;

      bool needs_refinement = false;

      Kokkos::parallel_reduce("DensityGradientCheck",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({is, js}, {ie + 1, je + 1}),
        KOKKOS_LAMBDA(int i, int j, bool& local_refine){
          if(rel_density_gradient(data, b, i, j) > refine_grad_threshold) local_refine = true;
        },
        Kokkos::LOr<bool>(needs_refinement)
      );
      return needs_refinement;
    };

    BlockCriterion density_gradient_derefine = [derefine_grad_threshold](const MeshBlock& block,
                                                                              const View4D& data,
                                                                              const View3D& rho_init){
      int b = block.slot_id;
      int is = block.is, ie = block.ie;
      int js = block.js, je = block.je;

      bool all_cells_quiet = true;

      Kokkos::parallel_reduce("DensityGradientDerefineCheck",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({is, js}, {ie + 1, je + 1}),
        KOKKOS_LAMBDA(int i, int j, bool& local_quiet){
          if(rel_density_gradient(data, b, i, j) >= derefine_grad_threshold) local_quiet = false;
        },
        Kokkos::LAnd<bool>(all_cells_quiet)
      );
      return all_cells_quiet;
    };

    ExchangeGhostZones(mesh);
    apply_bcs();
    auto data_check = mesh.get_data_view();
    int n_act = mesh.num_active_blocks();
    int nx1_b = mesh.inputs().nx1_block;
    int nx2_b = mesh.inputs().nx2_block;
    int ng = mesh.inputs().nghost;
    Real gamma = par.gamma;

    int n_invalid = 0;
    Kokkos::parallel_reduce("CheckStates",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_act, nx1_b + 2*ng, nx2_b + 2*ng}),
      KOKKOS_LAMBDA(int b, int i, int j, int& bad){
        Real rho = data_check(b, i, j, IDN);
        Real mom_u = data_check(b, i, j, IM1);
        Real mom_v = data_check(b, i, j, IM2);
        Real E = data_check(b, i, j, IEN);
        Real p = (gamma - 1.0)*(E - 0.5*(mom_u*mom_u + mom_v*mom_v)/(rho > 1e-12 ? rho : 1e-12));
        
        if (!(rho > 0.0) || !(p > 0.0)){
          //this will print the exact block slot and cell coordinates causing the failure!
          printf("INVALID STATE at block %d, cell (%d,%d): rho=%f, p=%f\n", b, i, j, rho, p);
          bad += 1;
        }
      }, n_invalid
    );
    if(n_invalid > 0){
      throw std::runtime_error("initial state has " + std::to_string(n_invalid) +
                               " cells (ghost cells included) with rho <= 0 or p <= 0: "
                               "check the initial condition and the boundary states");
    }
    Kokkos::fence();

    Real t0 = par.t0;
    Real tf = par.tf;
    Real Nt = par.Nt;
    Real dt = (tf - t0)/Nt;

    //CFL check
    Real cfl = compute_cfl_dt(mesh, par.gamma, par.cfl);
    if (dt > cfl) {
      std::cout << "Warning: Initial dt=" << dt << " exceeds CFL limit=" << cfl << "\n";
      std::cout << "Forcing dt to safe CFL limit for step 1...\n";
      dt = cfl; //automatically capping it so step 1 doesn't blow up!
    }

    Real t = t0;
    int n = 0;

    Real t_save = (tf - t0)/par.n_frames; 
    Real t_next_save = t0;

    //progressbar
    int lastpercentshown = -1;
    auto draw_progress = [&](Real t_now, bool force){
      int barwidth = 40;
      Real progress = (tf > par.t0) ? (t_now - par.t0)/(tf - par.t0) : 1.0;
      if (progress > 1.0) progress = 1.0;
      int percent = (int)(progress*100.0);
      if (!force && percent == lastpercentshown) return;
      lastpercentshown = percent;
      int filled = (int)(progress * barwidth);
      std::cout << "\r[" << std::string(filled, '#') << std::string(barwidth - filled, '-') << "] "
                << std::setw(3) << percent << "%  t=" << std::fixed << std::setprecision(3) << t_now
                << "  step " << n << std::flush;
    };

    //runtime measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    double cumulative_mass_loss = 0.0;
    double cumulative_energy_loss = 0.0;

    try{
      while(t < tf){
        //AMR part
        if(par.use_amr && n%par.amr_interval == 0){
          std::vector<std::unique_ptr<MeshBlock>> old_blocks;
          for(auto& b : mesh.blocks()) old_blocks.push_back(std::make_unique<MeshBlock>(*b));
          
          View4D old_data("old_data", inputs.max_blocks, nx1_b + 2*ng, nx2_b + 2*ng, NVAR);
          Kokkos::deep_copy(old_data, mesh.get_data_view());

          //run Tree Refinement & Derefinement
          mesh.UpdateDynamicMesh(density_gradient_refine, density_gradient_derefine, par.amr_max_level, derefine_grad_threshold);

          //(UpdateDynamicMesh already rebuilt the active block slots and neighbor maps)

          //map old states to new blocks (Direct, Prolongation, Restriction)
          mesh.TransferSolution(old_blocks, old_data);

          //populating ghost zones and boundaries for the new mesh layout immediately!!!
          //OTHERWISE UNSTABLE, DO NOT REMOVE
          ExchangeGhostZones(mesh);
          apply_bcs();
        }

        //CFL every step (also right after regridding!!! OTHERWISE UNSTABLE, DO NOT REMOVE)
        Real dt_cfl = compute_cfl_dt(mesh, par.gamma, par.cfl);
        if(!std::isfinite(dt_cfl) || dt_cfl <= 0.0) //a blown-up state would otherwise loop forever
          throw std::runtime_error("invalid time step dt = " + std::to_string(dt_cfl) +
                                  " at t = " + std::to_string(t) +
                                  ": the solution contains non-physical values (NaN, rho <= 0 or p <= 0)");
        dt = std::min(dt_cfl, tf - t);

        draw_progress(t, n == 0);

        if(t >= t_next_save){
          writeFrame(gridfile, mesh, t);
          gridfile.flush();
          t_next_save += t_save;
        }

        step_hydro_solver(mesh, dt, par.gamma, par.recon);
        ApplyReflux(mesh, dt);
        if(hooks.source_terms) hooks.source_terms(mesh, dt);
        ExchangeGhostZones(mesh);
        apply_bcs();

        //flux_view still holds this step's face fluxes (reflux and ghost exchange don't touch it)
        cumulative_mass_loss   += compute_boundary_outflow(mesh, 0) * dt;
        cumulative_energy_loss += compute_boundary_outflow(mesh, 3) * dt;

        n += 1;
        t += dt;
      }

      writeFrame(gridfile, mesh, t);
      gridfile.flush();
      draw_progress(tf, true);

      //runtime measurement
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;

      std::cout << "\nSimulation completed in " << n << " steps!\n";
      std::cout << "Output saved to " << par.outdir << "/grid.bin\n";

      double final_mass = compute_total(mesh, 0);
      double final_energy = compute_total(mesh, 3);
      
      double delta_mass_domain = std::abs(initial_mass - final_mass);
      double delta_energy_domain = std::abs(initial_energy - final_energy);

      double expected_mass = initial_mass - cumulative_mass_loss;
      double expected_energy = initial_energy - cumulative_energy_loss;

      std::cout << std::scientific << std::setprecision(12);
      std::cout << "  Initial Mass M(t=0)      = " << initial_mass << "\n";
      std::cout << "  Final Mass M(t=tf)       = " << final_mass << "\n";
      std::cout << "  Domain Drop |M0 - Mf|    = " << delta_mass_domain << "\n";
      std::cout << "  Net Boundary Loss (Flux) = " << std::abs(cumulative_mass_loss) << "\n";
      std::cout << "  Mass Discrepancy         = " << std::abs(final_mass - expected_mass) << "\n\n";

      std::cout << "  Initial Energy E(t=0)    = " << initial_energy << "\n";
      std::cout << "  Final Energy E(t=tf)     = " << final_energy << "\n";
      std::cout << "  Domain Drop |E0 - Ef|    = " << delta_energy_domain << "\n";
      std::cout << "  Net Boundary Loss (Flux) = " << std::abs(cumulative_energy_loss) << "\n";
      std::cout << "  Energy Discrepancy       = " << std::abs(final_energy - expected_energy) << "\n";
      std::cout << "RUNTIME: " << elapsed.count() << " seconds.\n";
    }catch(const std::exception& e){
      std::cerr << "\n[ERROR] Exception caught in simulation loop: " << e.what() << "\n";
      exit_code = 1;
    }
  }catch(const std::exception& e){
    std::cerr << "\n[ERROR] " << e.what() << "\n";
    exit_code = 1;
  }
  Kokkos::finalize();

  return exit_code;
}
