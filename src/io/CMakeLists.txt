#src/io: parameter file, output
file(GLOB IO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(szuchydro_io STATIC ${IO_SOURCES})
target_link_libraries(szuchydro_io PUBLIC szuchydro_mesh szuchydro_bvals szuchydro_hydro)
