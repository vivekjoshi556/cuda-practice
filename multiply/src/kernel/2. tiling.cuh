#include "../multiply.h"

/**
 * Implementation with Tiling & Shared Memory.
 * This expects block and tile size to be equal, so there is single tile per block.
 */
__global__ void multiplyV2(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int target_row = blockDim.y * blockIdx.y + threadIdx.y;
    int target_col = blockDim.x * blockIdx.x + threadIdx.x;
    
    int tx = target_col % TILE_WIDTH, ty = target_row % TILE_WIDTH;

    float result = 0;
    int ph_max = ceil(k/(TILE_WIDTH * 1.0));
    for(int ph = 0; ph < ph_max; ph++) { // This is called strip-mining approach.
        int offset = ph * TILE_WIDTH;
        if (offset + tx >= k || target_row >= m)
            Mds[ty][tx] = 0;
        else 
            Mds[ty][tx] = A[target_row * k + offset + tx];

        if (offset + ty >= k || target_col >= n)
            Nds[ty][tx] = 0;
        else 
            Nds[ty][tx] = B[(offset + ty) * n + target_col];
        
        // read-after-write dependence (True Dependence)
        // Threads must wait for data to be written to the proper place by other threads before reading it.
        // True dependence because thread truly needs the data supplied by writing thread, so it has no choice but to wait.
        __syncthreads(); 

        // Good for ILP:
        // Because memory loads are independent, so the loop can be unrolled and they can be made simultaneously.
        for(int k = 0; k < TILE_WIDTH; k++) {
            result += Mds[ty][k] * Nds[k][tx];
        }
        
        // write-after-read dependence (False Dependence)
        // Threads must wait for data to be read by all threads that need it before overwriting
        // False dependence because writing thread does not need any data from the reading thread.
        __syncthreads();
    }

    if (target_row < m && target_col < n) {
        C[target_row * n + target_col] = result;
    }
}
