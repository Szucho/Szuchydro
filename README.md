# Szuchydro

Szuchydro is a 2D finite-volume hydrodynamics code for astrophysical gas dynamics, written in C++20 with [Kokkos](https://github.com/kokkos/kokkos), so the same source runs on CPUs (OpenMP or serial) and NVIDIA GPUs (CUDA). It is developed for protoplanetary-disk and thermal-physics work, with an emphasis on exact hydrostatic equilibria and accurate temperatures.

> **Status:** under active development towards v1.0. Interfaces and parameter names may still change.

## Features

- **Hydrodynamics:** compressible Euler equations, HLLC Riemann solver, second-order reconstruction (limiters `minmod`, `vanleer`, `superbee`, `mc`; conserved or primitive variables).
- **Time integration:** `vl2` (van Leer predictor-corrector, default), `rk2` (Heun), `euler`.
- **Mesh:** block-structured static and adaptive mesh refinement (quadtree, 2:1 level balance, conservative refluxing). Refinement criteria: density gradient, pressure gradient, vorticity, contrast against the hydrostatic reference, or a problem-defined hook.
- **Well-balanced external potentials:** hydrostatic equilibria in any problem-defined potential stay at rest to machine precision: for any temperature profile in potentials varying along one axis (`wb_equilibrium linear`), and for isothermal or polytropic states in any 2D potential (`isothermal`, `polytropic`). The balance also holds across static refinement levels; AMR regrids are conservative.
- **Boundaries:** `Open`, `Closed`, `Periodic`, `Dirichlet`, `Outflow`, `Hydrostatic` (extends the equilibrium; impermeable reflecting wall).
- **Physics modules:** driven turbulence (few-mode Ornstein–Uhlenbeck forcing), linear drag, cooling to a set temperature.
- **Test suite:** regression tests with physics checks (exact solutions, conservation, symmetry, convergence order) and bit-for-bit comparison against saved references.

## Building

Requirements: CMake ≥ 3.22, a C++20 compiler (continuous integration uses GCC on Ubuntu 24.04), Python 3 with NumPy for the tests and scripts. Kokkos 5.0 is downloaded automatically (CMake FetchContent).

**CPU:**

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

`-DSZ_NATIVE=OFF` disables `-march=native` (for binaries that must run on other machines).

**NVIDIA GPU** (example: Ampere, compute capability 8.6; CUDA 13.2 or newer, since 13.0 conflicts with glibc 2.41 headers, yes I tried):

```bash
cmake -B build-gpu -S . -DCMAKE_BUILD_TYPE=Release -DSZ_NATIVE=OFF \
      -DCMAKE_CXX_COMPILER=$PWD/build/_deps/kokkos-src/bin/nvcc_wrapper \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON
cmake --build build-gpu -j "$(nproc)"
```

The `nvcc_wrapper` path comes from a CPU build directory, which downloads Kokkos first.

## Running

```bash
./build/szuchydro examples/sod_shock_tube.txt
```

A parameter file is a list of `key value` lines (`#` starts a comment). The problem is chosen with `gas_profile`; the grid with `Nx`, `Ny`, `xmin` … `ymax` and the block size `nx1_block`, `nx2_block`; boundaries with `bc_left`, `bc_right`, `bc_bottom`, `bc_top`; output with `outdir` and `n_frames`. Misspelled or unused keys are reported at start-up, and invalid values stop the run with a "Parameter error".

| `gas_profile` | Problem |
|---|---|
| `uniform` | uniform state |
| `Sod_x` | Sod shock tube |
| `sedov` | Sedov–Taylor blast wave |
| `disk_Rz` | stirred protoplanetary disk with winds in the R–z plane |
| `turb` | driven turbulence in a periodic box |
| `atmosphere` | stratified layer in hydrostatic equilibrium |
| `pointmass` | barotropic atmosphere around a softened point mass |
| `advect`, `vortex` | convergence tests (advected sine wave, isentropic vortex) |

The `examples/` folder has ready-to-run parameter files. Feel free to add new parameter files there using the existing problems or write an entirely new problem setup file (c++) in `src/pgen/your-new-problem.cpp`. The other files there will serve as guides, AI can also help in writing one.

## Output and analysis

Snapshots go to `<outdir>/grid.bin` (binary, all blocks with their geometry, might change to \textsc{HDF5} later) with a `meta.txt`. Python scripts in `scripts/`: animations (`animate.py`, `animate_lic.py`, `animate_meshblocks.py`), block layouts (`plot_meshblocks.py`), conservation checks (`conservation_check.py`) and kinetic-energy spectra with a Helmholtz decomposition (`spectrum.py`).

## Testing

```bash
python3 tests/run_tests.py --exe build/szuchydro                     # all tests
python3 tests/run_tests.py --exe build/szuchydro -k wb_               # tests whose name contains "wb_"
python3 tests/run_tests.py --exe build/szuchydro --save-ref ~/refs    # save references
python3 tests/run_tests.py --exe build/szuchydro --compare-ref ~/refs # bit-for-bit comparison
```

The same suite runs in continuous integration on every push.

## License and citation

To be added with the first release.
