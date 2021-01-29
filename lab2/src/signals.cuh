#ifndef SIGNALS_CUH
#define SIGNALS_CUH

#include <csignal>
#include <cstdlib>
#include <iostream>

namespace gpu {
void __nop_handler(int sig) {
    std::cout << "unexpected signal received: " << sig << std::endl;
    exit(0);
}

void handle_signals() {
    signal(SIGSEGV, __nop_handler);
    signal(SIGABRT, __nop_handler);
}
}  // namespace gpu

#endif