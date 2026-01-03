#ifndef OPTION_HPP
#define OPTION_HPP

struct OptionParameters {
    double S0;
    double K;
    double r;
    double sigma;
    double T;
};

inline double payoff(double ST, double K) {
    return (ST > K) ? (ST - K) : 0.0;
}

#endif
