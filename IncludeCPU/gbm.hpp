#ifndef GBM_HPP
#define GBM_HPP

#include <cmath>
#include "option.hpp"

struct GBrownianPrecompute {
    double drift;
    double diffusion;
    double discount;
};

inline GBrownianPrecompute precompute(const OptionParameters& opt) {
    return {
        (opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T,
        opt.sigma * std::sqrt(opt.T),
        std::exp(-opt.r * opt.T)
    };
}

inline double gbm_terminal_price(
    double S0,
    const GBrownianPrecompute& gbm,
    double Z
) {
    return S0 * std::exp(gbm.drift + gbm.diffusion * Z);
}

#endif
