#include "../multiply.h"

/**
 * Implementation with Tiling, Shared Memory & Warp Shuffle.
 * The idea here is rather than loading values to shared mem why not directly load in registers of each thread
 * and each time rather than loading that value from shared memory get it from the register, which should be faster.
 */
__global__ void multiplyV3(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int target_row = blockDim.y * blockIdx.y + threadIdx.y;
    int target_col = blockDim.x * blockIdx.x + threadIdx.x;

    int tx = target_col % TILE_WIDTH, ty = target_row % TILE_WIDTH;

    float result = 0;
    int ph_max = ceil(k/(TILE_WIDTH * 1.0));
    for(int ph = 0; ph < ph_max; ph++) { // This is called strip-mining approach.
        int offset = ph * TILE_WIDTH;
        Nds[tx][ty] = 0;
        float row_val = 0.0;

        if (offset + ty < k && target_row < m)
            row_val = A[target_row * k + offset + tx];
        
        if (offset + ty < k && target_col < n)
            Nds[tx][ty] = B[(offset + ty) * n + target_col];

        __syncthreads();
        
        #pragma unroll
        for(int i = 0; i < 32; i++) {
            // Bad for ILP:
            // In older version all the memory loads were coalesced if done together in a warp.
            //! Check: Here __shfl_sync will run 1 at a time (check if arch other than t4)
            //!! Also try removing the dependency chain and just __shfl_sync separately from FMA.
            result += __shfl_sync(FULL_MASK, row_val, i) * Nds[tx][i];
        }

        __syncthreads();
    }

    if (target_row < m && target_col < n) {
        C[target_row * n + target_col] = result;
    }
}
