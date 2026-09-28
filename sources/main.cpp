#include <cmath>
#include <iomanip>
#include <ios>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <limits>
#include <Kokkos_Core.hpp>

#include "hydro_config.hpp"
#include "grid.hpp"
#include "mesh.hpp"
#include "hllc.hpp"
#include "slope_limiters.hpp"
#include "hydro_io.hpp"

using namespace VecOps;

//CFL time step calculation (GPU/CPU Parallel Reduction)
Real compute_cfl_dt(Mesh& mesh, Real gamma, Real cfl_max){
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 1.0;

  View4D data = mesh.get_data_view();
  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;

  Kokkos::View<Real* [2], Kokkos::HostSpace> h_block_dx("h_block_dx", n_active);
  const auto& blocks = mesh.blocks();
  for (int b = 0; b < n_active; ++b) {
    h_block_dx(b, 0) = blocks[b]->dx1();
    h_block_dx(b, 1) = blocks[b]->dx2();
  }
  Kokkos::View<Real* [2]> d_block_dx("d_block_dx", n_active);
  Kokkos::deep_copy(d_block_dx, h_block_dx);

  Real max_wave_speed = 0.0;

  Kokkos::parallel_reduce(
    "ComputeCFL",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& local_max){
      Real dx1 = d_block_dx(b, 0);
      Real dx2 = d_block_dx(b, 1);

      Vec4 q_cell;
      for(size_t k = 0; k < NVAR; ++k){
        q_cell[k] = data(b, i, j, k);
      }

      Real rho = q_cell[0];
      Real u = q_cell[1]/rho;
      Real v = q_cell[2]/rho;
      Real p = (gamma - 1.0)*(q_cell[3] - rho*0.5*(u*u + v*v));
      Real cs = std::sqrt(gamma*p/rho);

      Real speed = (std::abs(u) + cs)/dx1 + (std::abs(v) + cs)/dx2;
      if(speed > local_max) local_max = speed;
    },
    Kokkos::Max<Real>(max_wave_speed)
  );

  return cfl_max/max_wave_speed;
}

//explicit 2D Euler Finite Volume timestep
void step_hydro_solver(Mesh& mesh, Real dt, Real gamma){
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

  Kokkos::View<Real* [2], Kokkos::HostSpace> h_block_dx("h_block_dx", n_active);
  const auto& blocks = mesh.blocks();
  for (int b = 0; b < n_active; ++b) {
    h_block_dx(b, 0) = (blocks[b]->block_size.x1max - blocks[b]->block_size.x1min) / blocks[b]->block_size.nx1;
    h_block_dx(b, 1) = (blocks[b]->block_size.x2max - blocks[b]->block_size.x2min) / blocks[b]->block_size.nx2;
  }
  Kokkos::View<Real* [2]> d_block_dx("d_block_dx", n_active);
  Kokkos::deep_copy(d_block_dx, h_block_dx);

  Kokkos::parallel_for(
    "StepHydroSolver",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){

      Real dx = d_block_dx(b, 0);
      Real dy = d_block_dx(b, 1);

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

      Vec4 sigma_im1_x = sigma_minmod(qim2, qim1, qi, dx);
      Vec4 sigma_i_x   = sigma_minmod(qim1, qi,   qip1, dx);
      Vec4 sigma_ip1_x = sigma_minmod(qi,   qip1, qip2, dx);

      Vec4 sigma_jm1_y = sigma_minmod(qjm2, qjm1, qi, dy);
      Vec4 sigma_j_y   = sigma_minmod(qjm1, qi,   qjp1, dy);
      Vec4 sigma_jp1_y = sigma_minmod(qi,   qjp1, qjp2, dy);

      Vec4 ql_imh, qr_imh, ql_iph, qr_iph;
      Vec4 ql_jmh, qr_jmh, ql_jph, qr_jph;

      for(size_t k = 0; k < NVAR; ++k){
        ql_imh[k] = qim1[k] + 0.5*dx*sigma_im1_x[k];
        qr_imh[k] = qi[k]   - 0.5*dx*sigma_i_x[k];
        ql_iph[k] = qi[k]   + 0.5*dx*sigma_i_x[k];
        qr_iph[k] = qip1[k] - 0.5*dx*sigma_ip1_x[k];

        ql_jmh[k] = qjm1[k] + 0.5*dy*sigma_jm1_y[k];
        qr_jmh[k] = qi[k]   - 0.5*dy*sigma_j_y[k];
        ql_jph[k] = qi[k]   + 0.5*dy*sigma_j_y[k];
        qr_jph[k] = qjp1[k] - 0.5*dy*sigma_jp1_y[k];
      }

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


//for conservation check
double compute_total_mass(Mesh& mesh) {
  double total_mass = 0.0;
  View4D data = mesh.get_data_view();
  for (const auto& blk_ptr : mesh.blocks()) {
    const auto& blk = *blk_ptr;
    int b = blk.slot_id;
    Real dx = blk.dx1();
    Real dy = blk.dx2();
    Real cell_vol = dx * dy;
    
    // Sum over active cell indices [is, ie] x [js, je]
    for (int i = blk.is; i <= blk.ie; ++i) {
      for (int j = blk.js; j <= blk.je; ++j) {
        total_mass += data(b, i, j, 0) * cell_vol; // rho is index 0
      }
    }
  }
  return total_mass;
}

double compute_total_energy(Mesh& mesh) {
  double total_energy = 0.0;
  View4D data = mesh.get_data_view();
for (const auto& blk_ptr : mesh.blocks()) {
    // const auto& blk_ptr = blk_ptr; // standard pointer
    const auto& blk = *blk_ptr;
    int b = blk.slot_id;
    Real dx = blk.dx1();
    Real dy = blk.dx2();
    Real cell_vol = dx * dy;
    
    for (int i = blk.is; i <= blk.ie; ++i) {
      for (int j = blk.js; j <= blk.je; ++j) {
        total_energy += data(b, i, j, 3) * cell_vol; // E is index 3
      }
    }
  }
  return total_energy;
}


int main(int argc, char* argv[]){
  Kokkos::initialize(argc, argv); //GPU accel
  {
    std::string param_file = (argc > 1) ? argv[1] : "params.txt"; //look for input file
    //reading and processing simulation parameters
    SimParams par = readParams(param_file);

    MeshInputs inputs;
    inputs.domain = RegionSize{par.xmin, par.xmax, par.ymin, par.ymax, (int)par.Nx, (int)par.Ny};
    inputs.nx1_block = par.nx1_block;
    inputs.nx2_block = par.nx2_block;
    inputs.nghost = par.nghost;
    inputs.max_blocks = 1024;
    inputs.refine_regions = par.refine_regions;

    GridBC bc;
    bc.left.type   = parseBC(par.bc_left);
    bc.right.type  = parseBC(par.bc_right);
    bc.bottom.type = parseBC(par.bc_bottom);
    bc.top.type    = parseBC(par.bc_top);
    validateBC(bc);

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

    init_cond(mesh, par);

    //capturing initial density profile
    mesh.CaptureInitialState();
    double initial_mass = compute_total_mass(mesh);
    double initial_energy = compute_total_energy(mesh);

    Real refine_grad_threshold = 0.05;   // Refine if local density gradient is high
    Real derefine_grad_threshold = 0.01; // Derefine if local gradients are quiet

    RefinementCriterion density_gradient_refine = [refine_grad_threshold](const MeshBlock& block,
                                                                          const View4D& data,
                                                                          const View3D& rho_init){
      int b = block.slot_id;
      int is = block.is, ie = block.ie;
      int js = block.js, je = block.je;

      bool needs_refinement = false;

      Kokkos::parallel_reduce("DensityGradientCheck",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({is, js}, {ie + 1, je + 1}),
        KOKKOS_LAMBDA(int i, int j, bool& local_refine){
          Real rho_ij = data(b, i, j, 0);
          // Check gradients with immediate neighbors
          Real rho_im = data(b, i - 1, j, 0);
          Real rho_ip = data(b, i + 1, j, 0);
          Real rho_jm = data(b, i, j - 1, 0);
          Real rho_jp = data(b, i, j + 1, 0);

          Real grad_x = std::abs(rho_ip - rho_im) / (2.0 * rho_ij + 1e-12);
          Real grad_y = std::abs(rho_jp - rho_jm) / (2.0 * rho_ij + 1e-12);

          if(grad_x > refine_grad_threshold || grad_y > refine_grad_threshold){
            local_refine = true;
          }
        },
        Kokkos::LOr<bool>(needs_refinement)
      );
      return needs_refinement;
    };

    DerefinementCriterion density_gradient_derefine = [derefine_grad_threshold](const MeshBlock& block,
                                                                              const View4D& data,
                                                                              const View3D& rho_init){
      int b = block.slot_id;
      int is = block.is, ie = block.ie;
      int js = block.js, je = block.je;

      bool all_cells_quiet = true;

      Kokkos::parallel_reduce("DensityGradientDerefineCheck",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({is, js}, {ie + 1, je + 1}),
        KOKKOS_LAMBDA(int i, int j, bool& local_quiet){
          Real rho_ij = data(b, i, j, 0);
          Real rho_im = data(b, i - 1, j, 0);
          Real rho_ip = data(b, i + 1, j, 0);
          Real rho_jm = data(b, i, j - 1, 0);
          Real rho_jp = data(b, i, j + 1, 0);

          Real grad_x = std::abs(rho_ip - rho_im) / (2.0 * rho_ij + 1e-12);
          Real grad_y = std::abs(rho_jp - rho_jm) / (2.0 * rho_ij + 1e-12);

          if(grad_x >= derefine_grad_threshold || grad_y >= derefine_grad_threshold){
            local_quiet = false;
          }
        },
        Kokkos::LAnd<bool>(all_cells_quiet)
      );
      return all_cells_quiet;
    };

    ExchangeGhostZones(mesh);
    ApplyPhysicalBoundaryConditions(mesh, bc);
    auto data_check = mesh.get_data_view();
    int n_act = mesh.num_active_blocks();
    int nx1_b = mesh.inputs().nx1_block;
    int nx2_b = mesh.inputs().nx2_block;
    int ng = mesh.inputs().nghost;

    Kokkos::parallel_for("CheckStates",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_act, nx1_b + 2*ng, nx2_b + 2*ng}),
      KOKKOS_LAMBDA(int b, int i, int j){
        Real rho = data_check(b, i, j, 0);
        Real mom_u = data_check(b, i, j, 1);
        Real mom_v = data_check(b, i, j, 2);
        Real E = data_check(b, i, j, 3);
        Real p = (1.4 - 1.0) * (E - 0.5 * (mom_u * mom_u + mom_v * mom_v) / (rho > 1e-12 ? rho : 1e-12));
        
        if (rho <= 0.0 || p <= 0.0){
          //this will print the exact block slot and cell coordinates causing the failure!
          printf("INVALID STATE at block %d, cell (%d,%d): rho=%f, p=%f\n", b, i, j, rho, p);
        }
      }
    );
    Kokkos::fence();

    Real t0 = par.t0;
    Real tf = par.tf;
    Real Nt = par.Nt;
    Real dt = (tf - t0)/Nt;

    //CFL check
    Real cfl = compute_cfl_dt(mesh, par.gamma, 0.4);
    if (dt > cfl) {
      std::cout << "Warning: Initial dt=" << dt << " exceeds CFL limit=" << cfl << "\n";
      std::cout << "Forcing dt to safe CFL limit for step 1...\n";
      dt = cfl; //automatically capping it so step 1 doesn't blow up!
    }

    Real t = t0;
    int n = 0;

    Real t_save = (tf - t0)/100.0; 
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
        if(n%5 == 0){
          std::vector<std::unique_ptr<MeshBlock>> old_blocks;
          for(auto& b : mesh.blocks()) old_blocks.push_back(std::make_unique<MeshBlock>(*b));
          
          View4D old_data("old_data", inputs.max_blocks, nx1_b + 2*ng, nx2_b + 2*ng, NVAR);
          Kokkos::deep_copy(old_data, mesh.get_data_view());

          //run Tree Refinement & Derefinement
          mesh.UpdateDynamicMesh(density_gradient_refine,
                                 density_gradient_derefine,
                                 /* max_level = */ 3, /* derefine_thresh = */ derefine_grad_threshold);

          //rebuild active block slots and block-neighbor maps
          mesh.CreateMeshBlocks();
          mesh.SetupNeighbors();

          //map old states to new blocks (Direct, Prolongation, Restriction)
          mesh.TransferSolution(old_blocks, old_data);

          //populating ghost zones and boundaries for the new mesh layout immediately!!!
          //OTHERWISE UNSTABLE, DO NOT REMOVE
          ExchangeGhostZones(mesh);
          ApplyPhysicalBoundaryConditions(mesh, bc);

          //also calculate cfl immediately!!! OTHERWISE UNSTABLE, DO NOT REMOVE
          Real dt_cfl_new = compute_cfl_dt(mesh, par.gamma, 0.4);
          dt = std::min(dt_cfl_new, tf - t);
        }

        // Real dt_cfl = compute_cfl_dt(mesh, par.gamma, 0.4);
        // dt = std::min(dt_cfl, tf - t);

        draw_progress(t, n == 0);

        if(t >= t_next_save){
          writeFrame(gridfile, mesh, t);
          gridfile.flush();
          t_next_save += t_save;
        }

        step_hydro_solver(mesh, dt, par.gamma);
        ApplyReflux(mesh, dt);
        ExchangeGhostZones(mesh);
        ApplyPhysicalBoundaryConditions(mesh, bc);

        for (const auto& blk_ptr : mesh.blocks()) {
          const auto& blk = *blk_ptr;
          int b = blk.slot_id;
          Real dx = blk.dx1();
          Real dy = blk.dx2();
          Real face_area = dy; // 2D strip area per cell along x-face

          auto data = mesh.get_data_view();

          // Check if block touches the left domain boundary (xmin)
          if (std::abs(blk.block_size.x1min - par.xmin) < 1e-12) {
              double left_mass_flux = 0.0;
              double left_energy_flux = 0.0;
              int i = blk.is; // Leftmost active cell index
              for (int j = blk.js; j <= blk.je; ++j) {
                  Real rho = data(b, i, j, 0);
                  Real mx  = data(b, i, j, 1);
                  Real E   = data(b, i, j, 3);
                  Real u   = mx / rho;
                  
                  left_mass_flux   += (rho * u) * face_area;
                  left_energy_flux += (E * u + (par.gamma - 1.0)*(E - 0.5*rho*u*u)*u) * face_area;
              }
              cumulative_mass_loss -= left_mass_flux * dt;
              cumulative_energy_loss -= left_energy_flux * dt;
          }

          // Check if block touches the right domain boundary (xmax)
          if (std::abs(blk.block_size.x1max - par.xmax) < 1e-12) {
              double right_mass_flux = 0.0;
              double right_energy_flux = 0.0;
              int i = blk.ie; // Rightmost active cell index
              for (int j = blk.js; j <= blk.je; ++j) {
                  Real rho = data(b, i, j, 0);
                  Real mx  = data(b, i, j, 1);
                  Real E   = data(b, i, j, 3);
                  Real u   = mx / rho;
                  
                  right_mass_flux   += (rho * u) * face_area;
                  right_energy_flux += (E * u + (par.gamma - 1.0)*(E - 0.5*rho*u*u)*u) * face_area;
              }
              cumulative_mass_loss += right_mass_flux * dt;
              cumulative_energy_loss += right_energy_flux * dt;
          }
        }

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

      double final_mass = compute_total_mass(mesh);
      double final_energy = compute_total_energy(mesh);
      
      double delta_mass_domain = std::abs(initial_mass - final_mass);
      double delta_energy_domain = std::abs(initial_energy - final_energy);

      double expected_mass = initial_mass - cumulative_mass_loss;
      double expected_energy = initial_energy - cumulative_energy_loss;

      std::cout << "\n==========================================================================\n";
      std::cout << "                 EXACT C++ CONSERVATION COMPARISON REPORT                 \n";
      std::cout << "==========================================================================\n";
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
      std::cout << "==========================================================================\n";
      std::cout << "RUNTIME: " << elapsed.count() << " seconds.\n";
    }catch(const std::exception& e){
      std::cerr << "\n[ERROR] Exception caught in simulation loop: " << e.what() << "\n";
    }
  }
  Kokkos::finalize();

  return 0;
}
