#include <chrono>
#include <iostream>
#include "multiply.h"

namespace chrono = std::chrono;

int main(int argc, char *argv[]) {
    bool use_user_input = false;
    if (argc > 1) {
        if (std::string(argv[1]) == "--input") {
            use_user_input = true;
        }
    }

    int NUM_THREADS  = 4;
    int m, n, k;
    std::cout << "Give Matrix Dimension: m, n, k for matrices (m, k) x (k, n):" << std::endl;
    std::cin >> m >> k >> n;
    std::cout << m << " " << n << " " << k << std::endl;

    matrix A(m, k), B(k, n), C(m, n), D(m, n);

    A.init(use_user_input);
    B.init(use_user_input);

    // std::cout << "CUDA Specs:" << std::endl;
    // getDeviceSpecs();
    // Serial Execution
    auto serialStart = chrono::high_resolution_clock::now();
    A.multiplySerial(&B, &C);
    auto serialTimer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - serialStart);
    printf("Time for Serial Execution: %.3f ms.\n", serialTimer.count());

    // Threaded Execution
    auto threadStart = chrono::high_resolution_clock::now();
    A.threadedMultiply(&B, &D, NUM_THREADS);
    auto threadTimer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - threadStart);

    if(D.matches(&C)) {
        printf("-------------------------------------------------------\n");
        printf("Time for Threded Execution %.3f ms.\n", threadTimer.count());
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/threadTimer));
    }
    else 
        printf("Threaded Execution returned incorrect result.\n");

    // Cuda Execution v1
    D.clear();
    auto cudaStart = chrono::high_resolution_clock::now();
    A.cudaMultiply(&B, &D, 1);
    auto cudaTimer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - cudaStart);

    printf("-------------------------------------------------------\n");
    if(D.matches(&C)) {
        printf("Time for Cuda Multiplication v1 %.3f ms.\n", cudaTimer.count());
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/cudaTimer));
        printf("SpeedUp over Threaded: x%.2f\n", (threadTimer/cudaTimer));
    }
    else
        printf("Cuda Execution v1 returned incorrect result.\n");

    // Cuda Execution v2
    D.clear();
    auto cudaV2Start = chrono::high_resolution_clock::now();
    A.cudaMultiply(&B, &D, 2);
    auto cudaV2Timer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - cudaV2Start);

    printf("-------------------------------------------------------\n");
    if(D.matches(&C)) {
        printf("Time for Cuda Multiplication v2 %.3f ms.\n", cudaV2Timer.count());
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/cudaV2Timer));
        printf("SpeedUp over Threaded: x%.2f\n", (threadTimer/cudaV2Timer));
        printf("SpeedUp over Cuda v1: x%.2f\n", (cudaTimer/cudaV2Timer));
    }
    else
        printf("Cuda Execution v2 returned incorrect result.\n");
    
    // Cuda Execution v3
    D.clear();
    auto cudaV3Start = chrono::high_resolution_clock::now();
    A.cudaMultiply(&B, &D, 3);
    auto cudaV3Timer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - cudaV3Start);

    printf("-------------------------------------------------------\n");
    if(D.matches(&C)) {
        printf("Time for Cuda Multiplication v3 %.3f ms.\n", cudaV3Timer.count());
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/cudaV3Timer));
        printf("SpeedUp over Threaded: x%.2f\n", (threadTimer/cudaV3Timer));
        printf("SpeedUp over Cuda v2: x%.2f\n", (cudaTimer/cudaV3Timer));
    }
    else
        printf("Cuda Execution v3 returned incorrect result.\n");

    // Cuda Execution v4
    D.clear();
    auto cudaV4Start = chrono::high_resolution_clock::now();
    A.cudaMultiply(&B, &D, 4);
    auto cudaV4Timer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - cudaV4Start);

    printf("-------------------------------------------------------\n");
    if(D.matches(&C)) {
        printf("Time for Cuda Multiplication v4 %.3f ms.\n", cudaV4Timer.count());
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/cudaV4Timer));
        printf("SpeedUp over Threaded: x%.2f\n", (threadTimer/cudaV4Timer));
        printf("SpeedUp over Cuda v2: x%.2f\n", (cudaTimer/cudaV4Timer));
    }
    else
        printf("Cuda Execution v4 returned incorrect result.\n");


    return 0;
}