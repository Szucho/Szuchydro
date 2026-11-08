// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "Kokkos_Version_Info.hpp"

namespace Kokkos {
namespace Impl {

std::string GIT_BRANCH       = R"branch()branch";
std::string GIT_COMMIT_HASH  = "";
std::string GIT_CLEAN_STATUS = "";
std::string GIT_COMMIT_DESCRIPTION =
    R"message()message";
std::string GIT_COMMIT_DATE = "";

}  // namespace Impl

}  // namespace Kokkos
