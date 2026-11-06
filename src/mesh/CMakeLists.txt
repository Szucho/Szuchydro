# src/mesh/CMakeLists.txt

#static library definition
add_library(szuchydro_mesh STATIC
  mesh.cpp
  meshtree.cpp
)

#includes are written relative to src/ (e.g. #include "core/hydro_config.hpp")
#PUBLIC, so everything linking the library gets the same include root
target_include_directories(szuchydro_mesh PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

#link Kokkos to the core library
target_link_libraries(szuchydro_mesh PUBLIC Kokkos::kokkos)
