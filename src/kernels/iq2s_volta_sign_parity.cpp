#include "strata/kernels/iq2s_volta.hpp"

#include <cstdint>
#include <cstdio>

namespace {

uint32_t reference(uint32_t packed, uint8_t bits) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane) {
        const uint8_t v = (uint8_t) (packed >> (lane * 8));
        const uint8_t r = (bits & (1u << lane)) ? (uint8_t) (0u - v) : v;
        out |= (uint32_t) r << (lane * 8);
    }
    return out;
}

bool check(uint32_t packed) {
    for (unsigned bits = 0; bits < 16; ++bits) {
        const uint32_t got = strata::kernels::detail::iq2s_apply_signs(packed, (uint8_t) bits);
        const uint32_t want = reference(packed, (uint8_t) bits);
        if (got != want) {
            std::fprintf(stderr, "iq2s_volta_sign_parity: packed=%08x bits=%x got=%08x want=%08x\n",
                         packed, bits, got, want);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    // Exhaust all non-zero byte values in every lane. A cross-byte carry can only
    // be introduced at a signed byte boundary, so varying each lane independently
    // is enough to pin the SWAR transform. Also test every combination of the
    // actual IQ2_S magnitudes used by the codebook.
    constexpr uint8_t fixed[4] = {8, 25, 43, 8};
    for (int lane = 0; lane < 4; ++lane) {
        for (unsigned v = 1; v <= 255; ++v) {
            uint32_t packed = 0;
            for (int j = 0; j < 4; ++j) {
                const uint8_t b = j == lane ? (uint8_t) v : fixed[j];
                packed |= (uint32_t) b << (j * 8);
            }
            if (!check(packed)) return 1;
        }
    }
    const uint8_t mag[3] = {8, 25, 43};
    for (uint8_t a : mag) for (uint8_t b : mag)
        for (uint8_t c : mag) for (uint8_t d : mag) {
            const uint32_t packed = (uint32_t) a | ((uint32_t) b << 8) |
                                    ((uint32_t) c << 16) | ((uint32_t) d << 24);
            if (!check(packed)) return 1;
        }
    std::puts("iq2s_volta_sign_parity: OK");
    return 0;
}
