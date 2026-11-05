cmake_minimum_required(VERSION 3.22)

#Silence FetchContent CMP0135 warning
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

#Prevent CMake from picking NMake by default on Windows if using MinGW
if(WIN32 AND NOT CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "MinGW Makefiles" CACHE INTERNAL "" FORCE)
endif()

project(Szuchydro-2D CXX)

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


add_subdirectory(mesh)

#sources will all be needed for the executable
file(GLOB SOURCES "sources/*.cpp")
add_executable(szuchydro2d ${SOURCES})

#link headers to mesh static library
target_include_directories(szuchydro2d PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/headers
)
target_link_libraries(szuchydro2d PRIVATE szuchydro_mesh)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/builds)

# add_executable(szuchydro2d ${SOURCES})

#Compiler Flag Optimizations (-march=native)
include(CheckCXXCompilerFlag)
option(SZ_NATIVE "Optimize for the CPU of build machine (-march=native)" ON)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if(SZ_NATIVE AND COMPILER_SUPPORTS_MARCH_NATIVE)
  target_compile_options(szuchydro2d PRIVATE
    $<$<CONFIG:Release>:-march=native>
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/params.txt")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/params.txt" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
  message(WARNING "params.txt not found in root directory! Make sure it is placed alongside CMakeLists.txt.")
endif()

#Link Kokkos library
target_link_libraries(szuchydro2d PRIVATE Kokkos::kokkos)
