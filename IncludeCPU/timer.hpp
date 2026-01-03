#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point start;

    Timer() : start(clock::now()) {}

    double elapsed() const {
        return std::chrono::duration<double>(
            clock::now() - start
        ).count();
    }
};

#endif
