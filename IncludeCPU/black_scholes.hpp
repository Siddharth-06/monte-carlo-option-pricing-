#ifndef BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_HPP

#include <cmath>

inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

inline double Black_Scholes_call(
    double S,
    double K,
    double r,
    double sigma,
    double T
) {
    double d1 =
        (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
        / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    return S * normal_cdf(d1)
        - K * std::exp(-r * T) * normal_cdf(d2);
}

#endif
