#include "MonteCarloMT.hpp"

#include "gbm.hpp"
#include "stats.hpp"
#include "option.hpp"

#include <random>
#include <thread>
#include <vector>
#include <cmath>

//
// -------- Worker (Antithetic + Control Variates) --------
//

void WorkerCV(
    const OptionParameters& opt,
    const GBrownianPrecompute& gbm,
    long long paths,
    unsigned long long seed,
    ControlStats& stats
) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double disc = gbm.discount;

    long long pairs = paths / 2;

    // Antithetic pairs
    for (long long i = 0; i < pairs; ++i) {
        double Z = normal(rng);

        double ST1 = gbm_terminal_price(opt.S0, gbm, Z);
        double ST2 = gbm_terminal_price(opt.S0, gbm, -Z);

        double f1 = disc * payoff(ST1, opt.K);
        double f2 = disc * payoff(ST2, opt.K);
        double f = 0.5 * (f1 + f2);

        double g1 = disc * ST1;
        double g2 = disc * ST2;
        double g = 0.5 * (g1 + g2);

        ControlUpdate(stats, f, g);
    }

    // Handle odd leftover path (if any)
    if (paths % 2 == 1) {
        double Z = normal(rng);
        double ST = gbm_terminal_price(opt.S0, gbm, Z);

        double f = disc * payoff(ST, opt.K);
        double g = disc * ST;

        ControlUpdate(stats, f, g);
    }
}

//
// -------- Multithreaded Monte Carlo --------
//

Result MonteCarloMultiThread(
    const OptionParameters& opt,
    long long N,
    int threads,
    unsigned long long seed
) {
    GBrownianPrecompute gbm = precompute(opt);

    std::vector<std::thread> pool;
    pool.reserve(threads);

    std::vector<ControlStats> states(threads);

    long long base = N / threads;
    long long rem = N % threads;

    for (int t = 0; t < threads; ++t) {
        long long paths = base + (t < rem ? 1 : 0);
        unsigned long long s = seed + 1337ULL * t;

        pool.emplace_back(
            WorkerCV,
            std::cref(opt),
            std::cref(gbm),
            paths,
            s,
            std::ref(states[t])
        );
    }

    for (auto& th : pool) th.join();

    // Merge all thread-local statistics
    ControlStats global;
    for (const auto& s : states)
        global = merge(global, s);

    // Control variate estimator
    double var_g = global.C_gg / (global.n - 1);
    double cov_fg = global.C_fg / (global.n - 1);
    double beta = cov_fg / var_g;

    double price =
        global.mean_f - beta * (global.mean_g - opt.S0);

    double var_cv =
        (global.C_ff
            - 2.0 * beta * global.C_fg
            + beta * beta * global.C_gg)
        / (global.n - 1);

    Result res;
    res.price = price;
    res.std_error = std::sqrt(var_cv / global.n);
    return res;
}
