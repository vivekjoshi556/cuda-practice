/**
 * Calculates 1 element per thread.
 */
__global__ void multiplyV1(float *A, float *B, float *C, int m, int n, int k) {
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    // Causes divergence at the ending rows and cols if the BLOCK_SIZE is not a factor of rows and cols.
    if (rowIdx >= m || colIdx >= n) {
        return;
    }

    float total = 0.0f;
    int idxA = rowIdx * k;
    int idxB = colIdx;
    for(int i = 0; i < n; i++, idxA++, idxB += n) {
        total += A[idxA] * B[idxB];
    }

    C[rowIdx * n + colIdx] = total;
}