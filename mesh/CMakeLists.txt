# mesh/CMakeLists.txt

#static library definition
add_library(szuchydro_mesh STATIC
  mesh.cpp
  meshtree.cpp
)

#find headers here
target_include_directories(szuchydro_mesh PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../headers
)

#link Kokkos to the core library
target_link_libraries(szuchydro_mesh PUBLIC Kokkos::kokkos)
