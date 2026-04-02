#include <assert.h>

/**
 * Implementation with Block 2D-Tiling.
 * The idea is to make each thread do more computation.
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void multiplyV5(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    float a_reg[TM] = {0.0}, b_reg[TN] = {0.0};
    float threadResult[TM * TN] = {0.0};

    const int numThreads = (BM * BN) / (TM * TN);

    const int ITEMS_PER_THREAD_X = BM / TM;

    const int threadIdxX = threadIdx.x % ITEMS_PER_THREAD_X;
    const int threadIdxY = threadIdx.x / ITEMS_PER_THREAD_X;

    const int aThreadRow = threadIdx.x / BK;
    const int aThreadCol = threadIdx.x % BK;
    const int strideA = numThreads / BK;

    const int bThreadRow = threadIdx.x / BN;
    const int bThreadCol = threadIdx.x % BN;
    const int strideB = numThreads / BN;

    for (int offset = 0; offset < k; offset += BK) {
        // Loading A into shared memory (BM, BK)
        for(uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(aThreadRow + loadOffset) * BK + aThreadCol] = A[(blockIdx.y * BM + loadOffset + aThreadRow) * k + offset + aThreadCol];
        }

        // Loading B into shared memory (BK, BN)
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(bThreadRow + loadOffset) * BN + bThreadCol] = B[(offset + bThreadRow + loadOffset) * n + blockIdx.x * BN + bThreadCol];
        }


        __syncthreads();

        // Remember each thread is doing this separately and there are (8x8) threads.
        for(int i = 0; i < BK; i++) {
            for(int j = 0; j < TM; j++) {
                a_reg[j] = As[(threadIdxY * TM + j) * BK + i];
            }

            for(int j = 0; j < TN; j++) {
                b_reg[j] = Bs[i * BN + threadIdxX * TN + j];
            }

            for(int j = 0; j < TM; j++) {
                for(int k = 0; k < TN; k++) {
                    threadResult[j * TN + k] += a_reg[j] * b_reg[k];
                }
            }
        }

        __syncthreads();
    }

    C += blockIdx.y * BM * n + blockIdx.x * BN;
    // Write out results
    for(int i = 0; i < TM; i++) {
        for(int j = 0; j < TN; j++) {
            C[(threadIdxY * TM + i) * n + threadIdxX * TN + j] = threadResult[i * TN + j];
        }
    }
}
