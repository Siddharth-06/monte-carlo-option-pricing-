#pragma once
#include <cmath>

struct GBrownianPrecompute {
    double drift;
    double diffusion;
    double discount;
};

__host__ __device__
inline GBrownianPrecompute precompute(const OptionParameters& opt) {
    return {
        (opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T,
        opt.sigma * sqrt(opt.T),
        exp(-opt.r * opt.T)
    };
}

__host__ __device__
inline double gbm_terminal_price(
    double S0,
    const GBrownianPrecompute& gbm,
    double Z
) {
    return S0 * exp(gbm.drift + gbm.diffusion * Z);
}
