#src/bvals: physical boundary conditions
file(GLOB BVALS_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(szuchydro_bvals STATIC ${BVALS_SOURCES})
target_link_libraries(szuchydro_bvals PUBLIC szuchydro_mesh)
