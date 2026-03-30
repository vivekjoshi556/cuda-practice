#include <stdio.h>

void getDeviceSpecs() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    printf("Number of Devices Available: %d\n", device_count);
    printf("--------------------------------------\n");

    for(int i = 0; i < device_count; i++) {
        printf("For Device: %d\n", i);
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, 0);

        printf("Max Threads Per Block: %d\n", dev_prop.maxThreadsPerBlock);
        printf("Warp Size: %d\n", dev_prop.warpSize);
        printf("Max Block Size: %d, %d, %d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
        printf("Max Grid Size: %d, %d, %d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
        printf("Register Per Block: %d\n", dev_prop.regsPerBlock);
        printf("\n");
        printf("Clock Rate: %d\n", dev_prop.clockRate);
        printf("Multi-Processor Count: %d\n", dev_prop.multiProcessorCount);
        printf("Register Per Multiprocessor: %d\n", dev_prop.regsPerMultiprocessor);
        printf("Max Blocks Per Multiprocessor: %d\n", dev_prop.maxBlocksPerMultiProcessor);
        printf("Max Threads Per Multiprocessor: %d\n", dev_prop.maxThreadsPerMultiProcessor);
        printf("Shared Memory Per Block: %zu\n", dev_prop.sharedMemPerBlock);
        printf("Shared Memory Per MultiProcessor: %zu\n", dev_prop.sharedMemPerMultiprocessor);
        printf("--------------------------------------\n");
    }
}