#include <iostream>
#include <thread>
#include <cmath>

#include "option.hpp"
#include "result.hpp"
#include "timer.hpp"
#include "MonteCarloST.hpp"
#include "MonteCarloMT.hpp"
#include "black_scholes.hpp"

int main() {
    OptionParameters opt{
        100.0,
        100.0,
        0.05,
        0.2,
        1.0
    };

    long long N = 10'000'000;   // push it
    unsigned long long seed = 42;

    std::cout << "Hardware threads: "
        << std::thread::hardware_concurrency()
        << "\n\n";

    // ---------- Single-thread ----------
    {
        Timer t;
        auto res = MonteCarloSingleThread(opt, N, seed);
        double time = t.elapsed();

        std::cout << "[ST]\n";
        std::cout << "Price     : " << res.price << "\n";
        std::cout << "Std Error : " << res.std_error << "\n";
        std::cout << "Time (ms)  : " << time*1000 << "\n\n";
    }

    // ---------- Multi-thread ----------
    {
        int threads = std::thread::hardware_concurrency();

        Timer t;
        auto res = MonteCarloMultiThread(opt, N, threads, seed);
        double time = t.elapsed();

        std::cout << "[MT - " << threads << " threads]\n";
        std::cout << "Price     : " << res.price << "\n";
        std::cout << "Std Error : " << res.std_error << "\n";
        std::cout << "Time (ms)  : " << time*1000 << "\n\n";
    }

    double bs = Black_Scholes_call(
        opt.S0, opt.K, opt.r, opt.sigma, opt.T
    );

    std::cout << "Black–Scholes: " << bs << "\n";
}
