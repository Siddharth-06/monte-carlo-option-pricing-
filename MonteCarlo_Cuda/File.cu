#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <chrono>

#include "option.hpp"
#include "gbm.hpp"
#include "stats.hpp"

__global__ void MonteCarloKernelCV(
    OptionParameters opt,
    GBrownianPrecompute gbm,
    ControlStats* out,
    unsigned long long seed,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);

    // Antithetic sampling
    double Z = curand_normal_double(&rng);

    double ST1 = gbm_terminal_price(opt.S0, gbm, Z);
    double ST2 = gbm_terminal_price(opt.S0, gbm, -Z);

    double f1 = gbm.discount * payoff(ST1, opt.K);
    double f2 = gbm.discount * payoff(ST2, opt.K);
    double f = 0.5 * (f1 + f2);

    double g1 = gbm.discount * ST1;
    double g2 = gbm.discount * ST2;
    double g = 0.5 * (g1 + g2);

    out[tid] = { f, g };
}

int main() {
    const int N = 1 << 20;

    OptionParameters opt{
        100.0, 100.0, 0.05, 0.2, 1.0
    };

    GBrownianPrecompute gbm = precompute(opt);

    ControlStats* d_out;
    cudaMalloc(&d_out, N * sizeof(ControlStats));

    int block = 256;
    int grid = (N + block - 1) / block;

    // ---------------- TIMER START ----------------
    auto t0 = std::chrono::high_resolution_clock::now();

    MonteCarloKernelCV << <grid, block >> > (
        opt,
        gbm,
        d_out,
        42ULL,
        N
        );
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    // ---------------- TIMER END ------------------

    // Copy back
    ControlStats* h_out = new ControlStats[N];
    cudaMemcpy(h_out, d_out, N * sizeof(ControlStats), cudaMemcpyDeviceToHost);

    // Host reduction
    double sum_f = 0.0, sum_g = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_f += h_out[i].f;
        sum_g += h_out[i].g;
    }

    double mean_f = sum_f / N;
    double mean_g = sum_g / N;

    // Control variate estimator (beta ≈ 1)
    double price = mean_f - (mean_g - opt.S0);

    // ---------- STANDARD ERROR ----------
    double var_f = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = h_out[i].f - mean_f;
        var_f += diff * diff;
    }
    var_f /= (N - 1);

    double std_error = std::sqrt(var_f / N);

    // Timing
    double time_s =
        std::chrono::duration<double>(t1 - t0).count();

    // Output
    printf("[GPU MC | Antithetic + CV]\n");
    printf("Price     : %.6f\n", price);
    printf("Std Error : %.6f\n", std_error);
    printf("95%% CI    : [%.6f, %.6f]\n",
        price - 1.96 * std_error,
        price + 1.96 * std_error);
    printf("Time (ms)  : %.6f\n", time_s*1000);

    cudaFree(d_out);
    delete[] h_out;
}
