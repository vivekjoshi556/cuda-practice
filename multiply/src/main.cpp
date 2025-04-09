#include <chrono>
#include <iostream>
#include "multiply.h"

namespace chrono = std::chrono;

int main() {
    int NUM_THREADS  = 4;
    int a, b, c, d;
    std::cout << "Give Matrix Dimension:" << std::endl;
    std::cin >> a >> b;

    matrix A(a, b), B(a, b), C(a, b), D(a, b), E(a, b), F(a, b);

    A.init();
    B.clone(&A);

    std::cout << "CUDA Specs:" << std::endl;
    getDeviceSpecs();
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
        printf("Time for Threded Execution %.3f ms.\n", threadTimer);
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/threadTimer));
    }
    else 
        printf("Threaded Execution returned incorrect result.\n");

    // Cuda Execution v1
    auto cudaStart = chrono::high_resolution_clock::now();
    A.cudaMultiplyV1(&B, &E);
    auto cudaTimer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - cudaStart);

    if(E.matches(&D)) {
        printf("-------------------------------------------------------\n");
        printf("Time for Cuda Multiplication v1 %.3f ms.\n", cudaTimer);
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/cudaTimer));
        printf("SpeedUp over Threaded: x%.2f\n", (threadTimer/cudaTimer));
    }
    else
        printf("Cuda Execution v1 returned incorrect result.\n");

    // Cuda Execution v2
    auto cudaV2Start = chrono::high_resolution_clock::now();
    A.cudaMultiplyV2(&A, &F);
    auto cudaV2Timer = chrono::duration<double, std::milli>(chrono::high_resolution_clock::now() - cudaV2Start);

    if(F.matches(&E)) {
        printf("-------------------------------------------------------\n");
        printf("Time for Cuda Multiplication v2 %.3f ms.\n", cudaV2Timer);
        printf("SpeedUp over Serial: x%.2f\n", (serialTimer/cudaV2Timer));
        printf("SpeedUp over Threaded: x%.2f\n", (threadTimer/cudaV2Timer));
        printf("SpeedUp over Cuda v1: x%.2f\n", (cudaTimer/cudaV2Timer));
    }
    else
        printf("Cuda Execution v2 returned incorrect result.\n");

    return 0;
}