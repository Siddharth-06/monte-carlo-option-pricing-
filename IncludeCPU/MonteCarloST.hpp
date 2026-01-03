#ifndef MONTECARLO_ST_HPP
#define MONTECARLO_ST_HPP

#include "option.hpp"
#include "result.hpp"

Result MonteCarloSingleThread(
    const OptionParameters& opt,
    long long N,
    unsigned long long seed
);

#endif
