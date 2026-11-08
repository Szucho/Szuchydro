# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-src")
  file(MAKE_DIRECTORY "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-src")
endif()
file(MAKE_DIRECTORY
  "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-build"
  "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix"
  "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix/tmp"
  "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp"
  "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix/src"
  "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/szuch/Szuchydro/build-gpu/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
