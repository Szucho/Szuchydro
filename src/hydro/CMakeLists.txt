#src/hydro: Riemann solver, limiters, reconstruction; header-only for now
add_library(szuchydro_hydro INTERFACE)
target_link_libraries(szuchydro_hydro INTERFACE szuchydro_core)
