/**
 * Implementation with Block 1D-Tiling.
 * The idea is to make each thread do more computation than just single output cell.
 */
template <int BM, int BN, int BK, int ITEMS_PER_THREAD>
__global__ void multiplyV4(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float accumulators[ITEMS_PER_THREAD] = {0.0f};

    // Right now we have 1D Blocks but we need that to calculate computation for a 2d block.
    // For, this we'll remap this 1D logically into a 2d structure.
    // There are 512 threads. So we'll map it similar to our B's tile shape, i.e., (BK, BN).
    // Now each thread is supposed to load 1 value for both A & B.

    int A_row_idx = threadIdx.x / BK;
    int A_col_idx = threadIdx.x % BK;
    int B_row_idx = threadIdx.x / BN;
    int B_col_idx = threadIdx.x % BN;
    float temp;

    for (int offset = 0; offset < k; offset += BK) {
        // First we load the data into the shared memory.
        As[A_row_idx * BK + A_col_idx] = (
            (blockIdx.y * BM + A_row_idx < m && offset + A_col_idx < k) ?
            A[(blockIdx.y * BM + A_row_idx) * k + offset + A_col_idx] : 
            0
        );
        Bs[B_row_idx * BN + B_col_idx] = (
            ((offset + B_row_idx) < k && blockIdx.x * BN + B_col_idx < n) ?
            B[(offset + B_row_idx) * n + blockIdx.x * BN + B_col_idx] :
            0
        );
        __syncthreads();

        // Each value is SOP of 8 values from As and Bs.
        for(int i = 0; i < BK; i++) {
            temp = Bs[i * BN + B_col_idx];
            // Each thread calculates 8 items.
            for(int j = 0; j < ITEMS_PER_THREAD; j++) {
                accumulators[j] += temp * As[(B_row_idx * ITEMS_PER_THREAD + j) * BK + i];
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        if((blockIdx.y * BM + B_row_idx * ITEMS_PER_THREAD + i) < m && blockIdx.x * BN + B_col_idx < n) {
            C[(blockIdx.y * BM + B_row_idx * ITEMS_PER_THREAD + i) * n + blockIdx.x * BN + B_col_idx] += accumulators[i];
        }
    }
}