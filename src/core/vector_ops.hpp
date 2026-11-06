#pragma once

#include <cstddef>
#include "core/hydro_config.hpp"

//fixed 4D vector for Hydro state Q = [rho, rho*u, rho*v, E]

//the usual vector operations that we will need
namespace VecOps{

  KOKKOS_INLINE_FUNCTION
  Vec4 add(const Vec4& a, const Vec4& b){
    return Vec4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 sub(const Vec4& a, const Vec4& b){
    return Vec4{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 mul(Real s, const Vec4& v){
    return Vec4{s * v[0], s * v[1], s * v[2], s * v[3]};
  }

} // namespace VecOps

//operators live next to Vec4 (global namespace) so they are found without "using namespace VecOps"
KOKKOS_INLINE_FUNCTION
Vec4 operator+(const Vec4& a, const Vec4& b){ return VecOps::add(a, b); }

KOKKOS_INLINE_FUNCTION
Vec4 operator-(const Vec4& a, const Vec4& b){ return VecOps::sub(a, b); }

KOKKOS_INLINE_FUNCTION
Vec4 operator*(Real s, const Vec4& v){ return VecOps::mul(s, v); }
