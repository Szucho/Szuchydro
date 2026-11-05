#include "hydro_io.hpp"
#include "Kokkos_CheckUsage.hpp"
#include "Kokkos_Macros.hpp"
#include "Serial/Kokkos_Serial_Parallel_Range.hpp"
#include "mesh.hpp"
#include <stdexcept>

/*
 * Here we handle input output
 *
 * parameter file parsing, boundary condition mapping, 
 * initial state setup via Kokkos device kernels,
 * simulation metadata logging, and binary grid/frame serialization.
*/

SimParams readParams(const std::string& filename){ //reads in params.txt
  std::ifstream f(filename);
  if(!f) throw std::runtime_error("Cannot open param file: " + filename);

  std::map<std::string, std::string> kv;
  std::vector<RefinementRegion> refine_regions;
  std::string line;
  while(std::getline(f, line)){ //file read in
    auto comment = line.find('#'); //# denotes comments
    if (comment != std::string::npos) line = line.substr(0, comment);
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::istringstream ss(line);
    std::string key; //we use keys in the input file
    ss >> key;
    if(key == "refine"){ //static refine
      RefinementRegion r;
      if(!(ss >> r.x1min >> r.x1max >> r.x2min >> r.x2max >> r.level))
        throw std::runtime_error("Malformed 'refine' line: " + line);
      refine_regions.push_back(r);
      continue;
    }
    std::string val;
    if(ss >> val) kv[key] = val;
  }

  auto has = [&](const std::string& k) { return kv.find(k) != kv.end(); };
  auto get = [&](const std::string& k) -> std::string{
      auto it = kv.find(k);
      if(it == kv.end()) throw std::runtime_error("Missing parameter: " + k);
      return it->second;
  };
  auto getd = [&](const std::string& k){ return std::stod(get(k)); };
  auto geti = [&](const std::string& k){ return std::stoi(get(k)); };

  SimParams p; //just reading in all the other parameters
  p.Nx   = getd("Nx");    p.Ny  = getd("Ny");   p.Nt  = getd("Nt");
  p.t0   = getd("t0");    p.tf  = getd("tf");
  p.xmin = getd("xmin");  p.xmax = getd("xmax");
  p.ymin = getd("ymin");  p.ymax = getd("ymax");
  //refinement parameters
  p.nx1_block = has("nx1_block") ? geti("nx1_block") : (int)p.Nx;
  p.nx2_block = has("nx2_block") ? geti("nx2_block") : (int)p.Ny;
  p.refine_regions = std::move(refine_regions);
  //physics
  p.gamma = getd("gamma");
  //BC-s
  p.bc_left   = get("bc_left");
  p.bc_right  = get("bc_right");
  p.bc_top    = get("bc_top");
  p.bc_bottom = get("bc_bottom");
  //gas initial condition
  p.gas_profile = get("gas_profile");
  p.gas_rho0    = getd("gas_rho0");
  p.gas_p0      = getd("gas_p0");
  p.gas_u0      = getd("gas_u0");
  p.gas_v0      = getd("gas_v0");
  p.outdir      = get("outdir");

  //optional keys (old params.txt files keep working this is for ppd)
  auto optd = [&](const std::string& k, Real def){ return has(k) ? getd(k) : def; };
  auto opti = [&](const std::string& k, int def){ return has(k) ? geti(k) : def; };
  p.n_frames = opti("n_frames", 100);
  p.use_amr  = opti("use_amr", 1) != 0;
  p.cfl      = optd("cfl", 0.4);
  p.sedov_E  = optd("sedov_E", p.sedov_E);
  p.sedov_r  = optd("sedov_r", p.sedov_r);
  DiskParams& d = p.disk;
  d.h_disk      = optd("h_disk", d.h_disk);
  d.h_wind      = optd("h_wind", d.h_wind);
  d.eta_tr      = optd("eta_tr", d.eta_tr);
  d.q_D         = optd("q_D", d.q_D);
  d.rho_floor   = optd("rho_floor", d.rho_floor);
  d.p_floor     = optd("p_floor", d.p_floor);
  d.beta_cool   = optd("beta_cool", d.beta_cool);
  d.stir_nmodes = opti("stir_nmodes", d.stir_nmodes);
  d.stir_amp    = optd("stir_amp", d.stir_amp);
  d.stir_tcorr  = optd("stir_tcorr", d.stir_tcorr);
  d.stir_damp   = optd("stir_damp", d.stir_damp);
  d.stir_lmin   = optd("stir_lmin", d.stir_lmin);
  d.stir_lmax   = optd("stir_lmax", d.stir_lmax);
  d.stir_seed   = (unsigned)opti("stir_seed", (int)d.stir_seed);
  d.wind_mach   = optd("wind_mach", d.wind_mach);
  d.wind_angle  = optd("wind_angle", d.wind_angle);
  d.wind_tau    = optd("wind_tau", d.wind_tau);
  d.corona_tau  = optd("corona_tau", d.corona_tau);
  

  return p;
}

//now that we have BC-s in the input, we want to use those as keys for the simulation
BCType parseBC(const std::string& s){
  if      (s == "Open")      return BCType::Open;
  else if (s == "Closed")    return BCType::Closed;
  else if (s == "Periodic")  return BCType::Periodic;
  else if (s == "Dirichlet") return BCType::Dirichlet;
  else if (s == "Outflow")   return BCType::Outflow;
  else throw std::invalid_argument("Unknown BC type: " + s);
}


//disk model with an |eta| table covering the whole domain incl. ghost cells
DiskModel BuildDiskModelFromParams(const SimParams& par){
  Real ymax_abs = std::max(std::abs(par.ymin), std::abs(par.ymax));
  Real eta_max = 1.5*ymax_abs/par.xmin + 1.0;
  return BuildDiskModel(par.disk, par.gas_rho0, eta_max);
}


//just some initial conditions
void init_cond(Mesh& mesh, const SimParams& par){
  View4D data = mesh.get_data_view();
  Real gamma = par.gamma;
  Real xmin = par.xmin;
  Real xmax = par.xmax;
  std::string profile = par.gas_profile;
  Real rho0 = par.gas_rho0;
  Real p0 = par.gas_p0;
  Real u0 = par.gas_u0;
  Real v0 = par.gas_v0;
  int nghost = mesh.inputs().nghost;

  //does disk init cond found in disk_phycics.cpp
  if(profile == "disk_Rz"){
    InitDiskRz(mesh, BuildDiskModelFromParams(par), gamma);
    return;
  }

  Real sedov_r_dep = 0.0, sedov_p_in = 0.0; //for sedov but no need to have it inside loop
  if(profile == "Sedov"){
    Real dx_base = (par.xmax - par.xmin)/par.Nx;
    sedov_r_dep = (par.sedov_r > 0.0) ? par.sedov_r : 3.0*dx_base; 
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
    sedov_p_in = (gamma - 1.0)*par.sedov_E/area;
  }

  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int b_slot = b.slot_id;
    int total_i = b.block_size.nx1 + 2*nghost;
    int total_j = b.block_size.nx2 + 2*nghost;

    if(profile == "uniform"){
      Kokkos::parallel_for("InitUniform", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
        KOKKOS_LAMBDA(int i, int j) {
          data(b_slot, i, j, 0) = rho0;
          data(b_slot, i, j, 1) = rho0*u0;
          data(b_slot, i, j, 2) = rho0*v0;
          data(b_slot, i, j, 3) = p0/(gamma - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);
        }
      );
    }else if(profile == "Sod_x"){
      Real rho_L = rho0,     p_L = p0,     u_L = 0.0, v_L = 0.0;
      Real rho_R = rho0/8.0, p_R = 0.1*p0, u_R = 0.0, v_R = 0.0;
      Real x_mid = 0.5 * (xmin + xmax);
      int is = b.is; //offset to map cell index to physical coordinates
      Real x1min = b.block_size.x1min;
      Real dx1 = b.dx1();

      //initialize across the ENTIRE block including ghosts so no cell is left with NaN/zeros
      Kokkos::parallel_for("InitSod", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
        KOKKOS_LAMBDA(int i, int j) {
          Real x = x1min + (i - is + 0.5)*dx1;
          Real rho = (x < x_mid) ? rho_L : rho_R;
          Real p   = (x < x_mid) ? p_L   : p_R;
          Real u   = (x < x_mid) ? u_L   : u_R;
          Real v   = (x < x_mid) ? v_L   : v_R;
          Real E   = p/(gamma - 1.0) + 0.5*rho*(u*u + v*v);

          data(b_slot, i, j, 0) = rho;
          data(b_slot, i, j, 1) = rho*u;
          data(b_slot, i, j, 2) = rho*v;
          data(b_slot, i, j, 3) = E;
        }
      );
    }else if(profile == "sedov"){
      //uniform ambient medium + energy sedov_E deposited inside r<r_dep (pressure p_in)
      int is = b.is, js = b.js;
      Real x1min = b.block_size.x1min, x2min = b.block_size.x2min;
      Real dx1 = b.dx1(), dx2 = b.dx2();
      Real xc = 0.5*(par.xmin + par.xmax), yc = 0.5*(par.ymin + par.ymax);
      Real r_dep = sedov_r_dep, p_in = sedov_p_in;
      Kokkos::parallel_for("InitSedov", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
        KOKKOS_LAMBDA(int i, int j){
          Real x = x1min + (i - is + 0.5)*dx1 - xc;  
          Real y = x2min + (j - js + 0.5)*dx2 - yc;
          Real p = (x*x + y*y < r_dep*r_dep) ? p0 + p_in : p0; //ambient + blast energy
          data(b_slot, i, j, 0) = rho0;
          data(b_slot, i, j, 1) = 0.0;
          data(b_slot, i, j, 2) = 0.0;
          data(b_slot, i, j, 3) = p/(gamma - 1.0);
        }
      );
    }else{
      throw std::runtime_error("Unknown gas_profile: " + profile);
    }
  }
}

//output handling
void setupOutputDir(const std::string& dir){
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw std::runtime_error("Could not create output directory: " + dir);
    std::cout << "Output directory: " << dir << "\n";
}

//metadata contains simulation parameters
void writeMetadata(const std::string& outdir, const SimParams& par){
    std::ofstream f(outdir + "/meta.txt");
    if(!f) throw std::runtime_error("Cannot open meta.txt");
    f << std::setprecision(17) << std::scientific;
    f << "xmin "  << par.xmin  << "\n"
      << "xmax "  << par.xmax  << "\n"
      << "ymin "  << par.ymin  << "\n"
      << "ymax "  << par.ymax  << "\n"
      << "gamma " << par.gamma << "\n"
      << "Nx "    << (int)par.Nx << "\n"
      << "Ny "    << (int)par.Ny << "\n";
}

//grid.bin contains simulation data
std::ofstream openGridFile(const std::string& outdir, Mesh& mesh){
  std::string path = outdir + "/grid.bin";
  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("Cannot open grid output file: " + path);
  return f;
}

//writing into grid.bin
void writeFrame(std::ofstream& f, Mesh& mesh, Real t){
  //write timestamp
  const double t_out = t; //t is Real but this way it will always be a double in write out
  f.write(reinterpret_cast<const char*>(&t_out), sizeof(double));

  //write active block count for this specific frame
  uint64_t nblocks = mesh.blocks().size();
  f.write(reinterpret_cast<const char*>(&nblocks), sizeof(uint64_t));

  //write block geometries for this frame
  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int32_t level = b.loc.level;
    int64_t lx1 = b.loc.lx1, lx2 = b.loc.lx2;
    uint64_t nx1 = (uint64_t)b.block_size.nx1, nx2 = (uint64_t)b.block_size.nx2;
    f.write(reinterpret_cast<const char*>(&level), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&lx1), sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(&lx2), sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(&b.block_size.x1min), sizeof(double));
    f.write(reinterpret_cast<const char*>(&b.block_size.x1max), sizeof(double));
    f.write(reinterpret_cast<const char*>(&b.block_size.x2min), sizeof(double));
    f.write(reinterpret_cast<const char*>(&b.block_size.x2max), sizeof(double));
    f.write(reinterpret_cast<const char*>(&nx1), sizeof(uint64_t));
    f.write(reinterpret_cast<const char*>(&nx2), sizeof(uint64_t));
  }

  //write cell data payload
  View4D device_data = mesh.get_data_view();
  HostView4D host_data = Kokkos::create_mirror_view(device_data);
  Kokkos::deep_copy(host_data, device_data);

  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    for(int i = b.is; i <= b.ie; i++){
      for(int j = b.js; j <= b.je; j++){
        double cell_data[NVAR];
        for(size_t k = 0; k < NVAR; ++k){
          cell_data[k] = host_data(b.slot_id, i, j, k);
        }
        f.write(reinterpret_cast<const char*>(cell_data), sizeof(double) * NVAR);
      }
    }
  }
}
