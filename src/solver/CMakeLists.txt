#src/solver: the right-hand side on the mesh (face fluxes, flux divergence) and the time integrators
file(GLOB SOLVER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(szuchydro_solver STATIC ${SOLVER_SOURCES})
target_link_libraries(szuchydro_solver PUBLIC szuchydro_mesh szuchydro_hydro szuchydro_bvals)
