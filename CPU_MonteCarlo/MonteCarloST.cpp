
#include "../IncludeCPU/MonteCarloST.hpp"

#include "../IncludeCPU/gbm.hpp"
#include "../IncludeCPU/stats.hpp"
#include "../IncludeCPU/option.hpp"

#include <random>
#include "../IncludeCPU/black_scholes.hpp"

Result MonteCarloSingleThread(
    const OptionParameters& opt,
    long long N,
    unsigned long long seed
) {
    GBrownianPrecompute gbm = precompute(opt);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    ControlStats stats;
    double disc = gbm.discount;

    for (long long i = 0; i < N / 2; ++i) {
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


    double var_g = stats.C_gg / (stats.n - 1);
    double cov_fg = stats.C_fg / (stats.n - 1);

    double beta = cov_fg / var_g;

    double price =
        stats.mean_f - beta * (stats.mean_g - opt.S0);

    double var_cv =
        (stats.C_ff
            - 2 * beta * stats.C_fg
            + beta * beta * stats.C_gg)
        / (stats.n - 1);

    Result res;
    res.price = price;
    res.std_error = std::sqrt(var_cv / stats.n);
    return res;

}
