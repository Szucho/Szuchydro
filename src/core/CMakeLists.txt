#src/core: basic types (Real, Vec4, views, variable indices); header-only
add_library(szuchydro_core INTERFACE)

#every module writes its includes relative to src/ (e.g. #include "hydro/hllc.hpp")
target_include_directories(szuchydro_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(szuchydro_core INTERFACE Kokkos::kokkos)

#-march=native for ALL project code: every target links szuchydro_core
if(SZ_NATIVE AND COMPILER_SUPPORTS_MARCH_NATIVE)
  target_compile_options(szuchydro_core INTERFACE $<$<CONFIG:Release>:-march=native>)
endif()
