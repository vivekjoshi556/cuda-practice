/**
 * Implementation with Block 2D-Tiling.
 * The idea is to make each thread do more computation.
 */
template <int BM, int BN, int BK, int ITEMS_PER_THREAD_X, int ITEMS_PER_THREAD_Y>
__global__ void multiplyV5(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    for (int offset = 0; offset < k; offset += BK) {
        // Loading A into shared memory (BM, BK)
        uint offset_idx = (blockIdx.y * BM + threadIdx.x) * k + offset * BK;
        for(uint i = 0; i < BK; i++) {
            As[threadIdx.x * BK + i] = A[offset_idx + i];
        }

        // Loading A into shared memory (BM, BK)
        offset_idx = offset * BK * n + blockIdx.x * BN + threadIdx.x;
        for(uint i = 0; i < BK; i++) {
            Bs[i * BN + threadIdx.x] = B[offset_idx + offset_idx + i * n];
        }

        __syncthreads();

        // Calculate per thread result.
        
    }

    // Write out results
}
