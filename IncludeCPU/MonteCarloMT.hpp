#ifndef MONTECARLO_MT_HPP
#define MONTECARLO_MT_HPP

#include "option.hpp"
#include "result.hpp"

Result MonteCarloMultiThread(
    const OptionParameters& opt,
    long long N,
    int threads,
    unsigned long long seed
);

#endif
