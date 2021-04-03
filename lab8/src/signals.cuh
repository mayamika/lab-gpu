#ifndef SIGNALS_CUH
#define SIGNALS_CUH

#include <csignal>
#include <iostream>

namespace signals {
void __nop_handler(int sig) {
    std::cerr << "Unexpected signal received: " << sig << std::endl;
    exit(0);
}

void HandleSignals() {
    std::signal(SIGSEGV, __nop_handler);
    std::signal(SIGABRT, __nop_handler);
}
}  // namespace signals

#endif