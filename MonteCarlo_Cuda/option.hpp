#pragma once
#include <cmath>

struct OptionParameters {
    double S0;
    double K;
    double r;
    double sigma;
    double T;
};
__host__ __device__
inline double payoff(double ST, double K) {
    return ST > K ? ST - K : 0.0;
}
