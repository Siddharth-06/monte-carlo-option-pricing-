# GPU-Accelerated Monte Carlo Option Pricing

This project implements Monte Carlo pricing of **European call options** using:

- CPU (single-threaded)
- CPU (multi-threaded)
- GPU (CUDA)

To improve statistical efficiency, **antithetic sampling** and **control variates** are used across all implementations.  
All results are validated against the **Black–Scholes closed-form solution**.

---

## 1. Mathematical Model

The underlying asset price follows **Geometric Brownian Motion (GBM)**:

$$
S_T = S_0 \exp\left((r - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\right),
\quad Z \sim \mathcal{N}(0,1)
$$

The discounted payoff of a European call option is:

$$
f = e^{-rT} \max(S_T - K, 0)
$$

---

## 2. Monte Carlo Estimation

The option price is estimated as the expectation:

$$
V = \mathbb{E}[f]
$$

Using Monte Carlo simulation:

$$
\hat{V} = \frac{1}{N} \sum_{i=1}^{N} f_i
$$

This estimator is unbiased but has variance that decays as \( O(1/\sqrt{N}) \).

---

## 3. Variance Reduction Techniques

### 3.1 Antithetic Sampling

For each standard normal draw \( Z \), its antithetic counterpart \( -Z \) is also evaluated.

The paired estimator is:

$$
\tilde{f} = \frac{f(Z) + f(-Z)}{2}
$$

This reduces variance by exploiting the negative correlation between \( f(Z) \) and \( f(-Z) \).

---

### 3.2 Control Variates

A control variate \( g \) is chosen such that:

- \( g \) is strongly correlated with the payoff \( f \)
- \( \mathbb{E}[g] \) is known exactly

In this project:

$$
g = e^{-rT} S_T, \quad \mathbb{E}[g] = S_0
$$

The control-variate estimator is:

$$
\hat{V}_{CV} = \bar{f} - \beta (\bar{g} - \mathbb{E}[g])
$$

where the optimal coefficient is:

$$
\beta = \frac{\operatorname{Cov}(f, g)}{\operatorname{Var}(g)}
$$

This can be interpreted as **linear regression of the payoff on the control variable**, subtracting predictable noise.

---

## 4. Implementations

### CPU (Single Thread)
- Sequential Monte Carlo loop
- Online Welford updates for mean and covariance
- High statistical efficiency per sample

### CPU (Multi Thread)
- Parallel sampling using `std::thread`
- Thread-local statistics merged safely
- Strong variance reduction with improved throughput

### GPU (CUDA)
- One Monte Carlo sample per thread
- Random number generation via `curand`
- Antithetic sampling and control variates applied per thread
- Final reduction and control-variate correction on host

---

## 5. Experimental Results (Fixed Time)

All methods use identical option parameters and variance-reduction techniques.

| Method | Time (ms) | Price | Std Error |
|------|----------|------|-----------|
| CPU – Single Thread | 286.77 | 10.4500 | 0.00087 |
| CPU – Multi Thread (32 threads) | 31.56 | 10.4503 | 0.00087 |
| GPU (Antithetic + CV) | **3.08** | 10.4471 | 0.00717 |
| Black–Scholes (Exact) | — | **10.4506** | — |

---

## 6. Key Observations

- The GPU achieves **orders-of-magnitude speedup** due to massive parallelism.
- CPU implementations achieve **lower variance per sample** due to more accurate estimation of control-variate coefficients.
- GPUs excel at **throughput**, while CPUs excel at **statistical efficiency**.
- When compared at equal wall-clock time, a clear **accuracy–performance tradeoff** emerges.

---

## 7. Conclusion

This project demonstrates that:

> GPUs dominate short-horizon Monte Carlo estimation due to sample throughput,  
> while CPUs outperform at longer runtimes due to superior variance-reduction efficiency.

The crossover point depends on the time budget and the effectiveness of variance-reduction techniques.

---

## 8. Technologies Used

- C++17
- CUDA
- CURAND
- Multithreading (`std::thread`)
- Monte Carlo methods
- Variance reduction (antithetic sampling, control variates)

---

## 9. Future Work

- Multi-stage GPU reductions for improved control-variate estimation
- Quasi-Monte Carlo (Sobol sequences)
- Extension to path-dependent options
- Performance comparison across different GPU architectures
