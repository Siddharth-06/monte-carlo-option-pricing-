#ifndef STATS_HPP
#define STATS_HPP

#include <cmath>

//
// ---------- BASIC WELFORD (optional, still useful) ----------
//

struct WelfordState {
    double mean = 0.0;
    double Msquared = 0.0;
    long long n = 0;
};

inline void WelfordUpdate(WelfordState& s, double x) {
    s.n++;
    double delta = x - s.mean;
    s.mean += delta / s.n;
    double delta2 = x - s.mean;
    s.Msquared += delta * delta2;
}

inline double variance(const WelfordState& s) {
    return s.Msquared / (s.n - 1);
}

inline double standard_error(const WelfordState& s) {
    return std::sqrt(variance(s) / s.n);
}

//
// ---------- CONTROL VARIATE STATISTICS ----------
//

struct ControlStats {
    long long n = 0;
    double mean_f = 0.0;
    double mean_g = 0.0;
    double C_ff = 0.0;
    double C_gg = 0.0;
    double C_fg = 0.0;
};

inline void ControlUpdate(
    ControlStats& s,
    double f,
    double g
) {
    s.n++;

    double df = f - s.mean_f;
    double dg = g - s.mean_g;

    s.mean_f += df / s.n;
    s.mean_g += dg / s.n;

    s.C_ff += df * (f - s.mean_f);
    s.C_gg += dg * (g - s.mean_g);
    s.C_fg += df * (g - s.mean_g);
}

//
// ---------- MERGE CONTROL STATS (FOR MULTITHREADING) ----------
//

inline ControlStats merge(
    const ControlStats& a,
    const ControlStats& b
) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;

    ControlStats out;
    out.n = a.n + b.n;

    double df = b.mean_f - a.mean_f;
    double dg = b.mean_g - a.mean_g;

    out.mean_f = a.mean_f + df * (double(b.n) / out.n);
    out.mean_g = a.mean_g + dg * (double(b.n) / out.n);

    out.C_ff = a.C_ff + b.C_ff + df * df * a.n * b.n / out.n;
    out.C_gg = a.C_gg + b.C_gg + dg * dg * a.n * b.n / out.n;
    out.C_fg = a.C_fg + b.C_fg + df * dg * a.n * b.n / out.n;

    return out;
}

#endif // STATS_HPP
