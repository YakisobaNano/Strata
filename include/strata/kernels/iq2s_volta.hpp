#pragma once
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_IQ2S_HD __host__ __device__ __forceinline__
#else
#define STRATA_IQ2S_HD inline
#endif

namespace strata::kernels::detail {

// IQ2_S Volta sign application. Each bit conditionally negates one byte in a packed
// four-byte codebook word. IQ2_S codebook bytes are non-zero, so adding 1 to a
// complemented signed byte cannot carry into the neighboring byte.
STRATA_IQ2S_HD uint32_t iq2s_sign_mask4(uint8_t bits) {
    const uint32_t b0 = 0u - (uint32_t) ( bits       & 1u);
    const uint32_t b1 = 0u - (uint32_t) ((bits >> 1) & 1u);
    const uint32_t b2 = 0u - (uint32_t) ((bits >> 2) & 1u);
    const uint32_t b3 = 0u - (uint32_t) ((bits >> 3) & 1u);
    return (b0 & 0x000000ffu) | (b1 & 0x0000ff00u) |
           (b2 & 0x00ff0000u) | (b3 & 0xff000000u);
}

STRATA_IQ2S_HD uint32_t iq2s_apply_signs(uint32_t packed_grid, uint8_t bits) {
    const uint32_t mask = iq2s_sign_mask4(bits);
    return (packed_grid ^ mask) + (mask & 0x01010101u);
}

}  // namespace strata::kernels::detail

#undef STRATA_IQ2S_HD
