cmake_minimum_required(VERSION 3.22)

#Silence FetchContent CMP0135 warning
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

#Prevent CMake from picking NMake by default on Windows if using MinGW
if(WIN32 AND NOT CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "MinGW Makefiles" CACHE INTERNAL "" FORCE)
endif()

project(Szuchydro CXX)

#Set C++ standard (Kokkos 5.0 requires C++20 minimum)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#Enable compile_commands.json generation for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Default to Release build if none specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#Fetch and configure Kokkos 5.0
include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Kokkos)


#mesh has its own static library
add_subdirectory(src/mesh)
#everything else in src/ goes into the executable
#CONFIGURE_DEPENDS: new .cpp files are picked up automatically,
#no need to re-run cmake
file(GLOB SOURCES CONFIGURE_DEPENDS
  "src/*.cpp"
  "src/bvals/*.cpp"
  "src/hydro/*.cpp"
  "src/io/*.cpp"
  "src/physics/*.cpp"
  "src/pgen/*.cpp"
)
add_executable(szuchydro ${SOURCES})

#includes are written relative to src/, e.g. #include "hydro/hllc.hpp"
target_include_directories(szuchydro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(szuchydro PRIVATE szuchydro_mesh)

include(CheckCXXCompilerFlag)
option(SZ_NATIVE "Optimize for the CPU of build machine (-march=native)" ON)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if(SZ_NATIVE AND COMPILER_SUPPORTS_MARCH_NATIVE)
  target_compile_options(szuchydro PRIVATE
    $<$<CONFIG:Release>:-march=native>
  )
endif()

#Link Kokkos library
target_link_libraries(szuchydro PRIVATE Kokkos::kokkos)
