#include "io/hydro_io.hpp"
#include "io/param_store.hpp"
#include "mesh/mesh.hpp"
#include <stdexcept>
#include <string>

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

  ParamStore store; //every "key value" line; problem generators read their own keys from it
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
    if(ss >> val) store.set(key, val);
  }

  //shorthands: the store checks the values ("Nx abc" -> clear error) and records what was read
  auto has  = [&](const std::string& k){ return store.has(k); };
  auto get  = [&](const std::string& k){ return store.get(k); };
  auto getd = [&](const std::string& k){ return store.get_real(k); };
  auto geti = [&](const std::string& k){ return store.get_int(k); };


  SimParams p; //just reading in all the other parameters
  p.Nx   = geti("Nx");    p.Ny  = geti("Ny");   p.Nt  = geti("Nt");
  p.t0   = getd("t0");    p.tf  = getd("tf");
  p.xmin = getd("xmin");  p.xmax = getd("xmax");
  p.ymin = getd("ymin");  p.ymax = getd("ymax");
  //refinement parameters
  p.nx1_block = has("nx1_block") ? geti("nx1_block") : p.Nx;
  p.nx2_block = has("nx2_block") ? geti("nx2_block") : p.Ny;
  p.refine_regions = std::move(refine_regions);
  //physics
  p.gamma = getd("gamma");
  //BC-s
  p.bc_left   = get("bc_left");
  p.bc_right  = get("bc_right");
  p.bc_top    = get("bc_top");
  p.bc_bottom = get("bc_bottom");
  //problem to set up (src/pgen/) 
  //its own parameters (gas_rho0, ...) are read by the problem generator
  p.gas_profile = get("gas_profile");
  p.outdir      = get("outdir");

  //optional keys (old params.txt files keep working this is for ppd)
  auto optd = [&](const std::string& k, Real def){ return has(k) ? getd(k) : def; };
  auto opti = [&](const std::string& k, int def){ return has(k) ? geti(k) : def; };
  p.n_frames = opti("n_frames", 100);
  p.use_amr  = opti("use_amr", 1) != 0;
  p.cfl      = optd("cfl", 0.4);
  p.max_blocks = opti("max_blocks", p.max_blocks);
  p.amr_max_level = opti("amr_max_level", p.amr_max_level);
  p.amr_interval = opti("amr_interval", p.amr_interval);
  p.amr_refine_tol = optd("amr_refine_tol", p.amr_refine_tol);
  p.amr_derefine_tol = optd("amr_derefine_tol", p.amr_derefine_tol);
  
  //reconstruction options
  const std::string lim = has("limiter") ? get("limiter") : "minmod";
  if     (lim == "minmod")   p.recon.limiter = LIM_MINMOD;
  else if(lim == "vanleer")  p.recon.limiter = LIM_VANLEER;
  else if(lim == "superbee") p.recon.limiter = LIM_SUPERBEE;
  else if(lim == "mc")       p.recon.limiter = LIM_MC;
  else throw std::runtime_error("Parameter error: unknown limiter '" + lim + "' (minmod, vanleer, superbee, mc)");
  const std::string rec = has("reconstruct") ? get("reconstruct") : "conserved";
  if     (rec == "conserved") p.recon.primitive = false;
  else if(rec == "primitive") p.recon.primitive = true;
  else throw std::runtime_error("Parameter error: unknown reconstruct '" + rec + "' (conserved, primitive)");

  //Dirichlet wall states, only for Dirichlet BC and not allowed for any other
  auto read_wall = [&](const std::string& wall, const std::string& type, SimParams::WallState& w){
    const char* comp[4] = {"_rho", "_u", "_v", "_p"};
    Real* dst[4] = {&w.rho, &w.u, &w.v, &w.p};
    for(int c = 0; c < 4; ++c){
      const std::string key = "bc_" + wall + comp[c];
      if(type == "Dirichlet"){
        if(!has(key)){ 
          throw std::runtime_error("Parameter error: bc_" + wall + " is Dirichlet but " + key + " is missing");
        }
        *dst[c] = getd(key);
      }else if(has(key)){
        throw std::runtime_error("Parameter error: " + key + 
                                  " is set but bc_" + wall + " is "
                                  + type + ", not Dirichlet");
      }
    }
    if(type == "Dirichlet" && (w.rho <= 0.0 || w.p <= 0.0)){
      throw std::runtime_error("Parameter error: bc_" + wall + "_rho and bc_" + wall + "_p must be positive");
    }
  };

  read_wall("left",   p.bc_left,   p.dir_left);
  read_wall("right",  p.bc_right,  p.dir_right);
  read_wall("bottom", p.bc_bottom, p.dir_bottom);
  read_wall("top",    p.bc_top,    p.dir_top);

  //sanity checks
  auto require = [](bool ok, const std::string& msg){
    if(!ok) throw std::runtime_error("Parameter error: " + msg);
  };
  require(p.Nx > 0 && p.Ny > 0, "Nx and Ny must be larger than 0");
  require(p.Nt > 0, "Nt must be larger than 0");
  require(p.max_blocks >= 1, "max_blocks must be at least 1");
  require(p.amr_max_level >= 1, "amr_max_level must be at least 1");
  require(p.amr_interval >= 1, "amr_interval must be at least 1");
  require(p.amr_derefine_tol < p.amr_refine_tol, "amr_derefine_tol must be smaller than amr_refine_tol (otherwise blocks flip-flop)");
  if(p.use_amr){
    for(const auto& r : p.refine_regions){
      require(r.level <= p.amr_max_level, "a 'refine' line asks for level " + std::to_string(r.level)  + 
                                          " but amr_max_level is " + std::to_string(p.amr_max_level));
    }
  }

  p.store = std::move(store); //problem generators read their keys from here
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
      << "Nx "    << par.Nx << "\n"
      << "Ny "    << par.Ny << "\n";
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


//Dirichlet walls: convert the primitive wall states from the parameter file to the
//conserved ghost-cell values (other walls keep Q_fixed = 0, which is never used)
void setDirichletStates(GridBC& bc, const SimParams& par){
  auto set = [&](BoundaryCondition& w, const SimParams::WallState& s){
    if(w.type == BCType::Dirichlet) w.Q_fixed = prim_to_cons(Vec4{s.rho, s.u, s.v, s.p}, par.gamma);
  };
  set(bc.left,   par.dir_left);
  set(bc.right,  par.dir_right);
  set(bc.bottom, par.dir_bottom);
  set(bc.top,    par.dir_top);
}
