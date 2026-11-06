#src/pgen: problem generators, one file per problem.
#OBJECT library: every object file is always linked, so no problem can be silently dropped
# (a STATIC library would leave out files that nothing references directly as it turns out)
file(GLOB PGEN_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(szuchydro_pgen OBJECT ${PGEN_SOURCES})
target_link_libraries(szuchydro_pgen PUBLIC szuchydro_io szuchydro_physics szuchydro_solver)
