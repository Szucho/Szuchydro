#src/mesh: block tree, ghost exchange, prolongation/restriction, refluxing
file(GLOB MESH_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(szuchydro_mesh STATIC ${MESH_SOURCES})
#hydro: prolongation uses the slope limiters
target_link_libraries(szuchydro_mesh PUBLIC szuchydro_core szuchydro_hydro)
