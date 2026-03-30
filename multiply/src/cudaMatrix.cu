#include <stdio.h>
#include "multiply.h"
#include "kernel/kernels.cuh"


void matrix::cudaMultiply(matrix *B, matrix *C, int kernel_idx) {
    if (this->cols != B->rows) {
        printf("Inconsistent shapes for matrix multiplication. matrixA.cols should be equal to matrixB.rows");
        return;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, this->rows * this->cols * sizeof(float));
    cudaMalloc((void **)&d_B, B->rows * B->cols * sizeof(float));
    cudaMalloc((void **)&d_C, C->rows * C->cols * sizeof(float));

    cudaMemcpy(d_A, this->data, this->rows * this->cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, B->rows * B->cols * sizeof(float), cudaMemcpyHostToDevice);

    // Try fixing this for BLOCK_SIZE 64
    dim3 dimGrid(
        ceil(this->cols / (BLOCK_SIZE * 1.0)),
        ceil(this->rows / (BLOCK_SIZE * 1.0))
    );
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    switch (kernel_idx) {
        case 1: 
            multiplyV1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, C->rows, C->cols, B->rows);
            break;
        case 2: 
            multiplyV2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, C->rows, C->cols, B->rows);
            break;
        case 3: 
            multiplyV3<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, C->rows, C->cols, B->rows);
            break;
        case 4: {
            const int ITEMS_PER_THREAD = 8;
            const int BM = 64;
            const int BN = 64;
            const int BK = 8;

            dim3 blockTilingDimGrid(
                ceil(C->cols / (BN * 1.0)),
                ceil(C->rows / (BM * 1.0))
            );
            dim3 blockTilingDimBlock(
                BM * BN / ITEMS_PER_THREAD
            );
            multiplyV4<BM, BN, BK, ITEMS_PER_THREAD>
                <<<blockTilingDimGrid, blockTilingDimBlock>>>(d_A, d_B, d_C, C->rows, C->cols, B->rows);

            break;
        }
        case 5: {
            const int ITEMS_PER_THREAD_X = 8;
            const int ITEMS_PER_THREAD_Y = 8;
            const int BM = 64;
            const int BN = 64;
            const int BK = 8;

            dim3 blockTilingDimGrid(
                ceil(C->cols / (BN * 1.0)),
                ceil(C->rows / (BM * 1.0))
            ); // 4096 / 64 => 64, 64
            dim3 blockTilingDimBlock(
                (BM * BN) / (ITEMS_PER_THREAD_X * ITEMS_PER_THREAD_Y)
            ); // (64 x 64) / (8 x 8) => (64, 1)
            multiplyV5<BM, BN, BK, ITEMS_PER_THREAD_X, ITEMS_PER_THREAD_Y>
                <<<blockTilingDimGrid, blockTilingDimBlock>>>(d_A, d_B, d_C, C->rows, C->cols, B->rows);

            break;
        }
        default:
            printf("Invalid Kernel Number: %d", kernel_idx);
            break;
    }

    cudaMemcpy(C->data, d_C, this->rows * this->cols * sizeof(float), cudaMemcpyDeviceToHost);
}