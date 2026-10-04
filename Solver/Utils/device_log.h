#pragma once

#include <utility>
#include <luisa/dsl/stmt.h>

#ifndef LCS_ENABLE_DEVICE_LOG
#define LCS_ENABLE_DEVICE_LOG 1
#endif

namespace lcs {

// Vulkan's native SPIR-V path cannot print yet. Suppressing diagnostic printf
// keeps kernels on that path without suppressing device_assert or health flags.
template<typename Fmt, typename... Args>
inline void solver_device_log(Fmt&& fmt, Args&&... args) noexcept {
#if LCS_ENABLE_DEVICE_LOG
    luisa::compute::device_log(std::forward<Fmt>(fmt), std::forward<Args>(args)...);
#endif
}

}
