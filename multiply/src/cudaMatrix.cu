#include <stdio.h>
#include "multiply.h"

__global__ void multiplyV1(int *A, int *B, int *C, int r, int c) {
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx >= r || colIdx >= c) {
        return;
    }

    int total = 0;
    int idxA = rowIdx * c;
    int idxB = colIdx;
    for(int i = 0; i < c; i++, idxA++, idxB += c) {
        total += A[idxA] * B[idxB];
    }

    C[rowIdx * c + colIdx] = total;
}

void matrix::cudaMultiplyV1(matrix *B, matrix *C) {
    if (this->cols != B->rows) {
        printf("Inconsistent shapes for matrix multiplication. matrixA.cols should be eequal to matrixB.rows");
        return;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, this->rows * this->cols * sizeof(int));
    cudaMalloc((void **)&d_B, B->rows * B->cols * sizeof(int));
    cudaMalloc((void **)&d_C, C->rows * C->cols * sizeof(int));

    cudaMemcpy(d_A, this->data, this->rows * this->cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, B->rows * B->cols * sizeof(int), cudaMemcpyHostToDevice);

    // Try fixing this for BLOCK_SIZE 64
    dim3 dimGrid(ceil(this->rows / (BLOCK_SIZE * 1.0)), ceil(this->rows / (BLOCK_SIZE * 1.0)));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiplyV1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, C->rows, C->cols);

    cudaMemcpy(C->data, d_C, this->rows * this->cols * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void multiplyV2(int *A, int *B, int *C, int r, int c) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;

    int target_row = blockDim.y * blockIdx.y + threadIdx.y;
    int target_col = blockDim.x * blockIdx.x + threadIdx.x;

    float result = 0;
    int ph_max = ceil(c/(TILE_WIDTH * 1.0));
    for(int ph = 0; ph < ph_max; ph++) { // This is called strip-mining approach.
        int offset = ph * TILE_WIDTH;
        if (offset + tx >= c || target_row >= r)
            Mds[ty][tx] = 0;
        else 
            Mds[ty][tx] = A[target_row * c + offset + tx];

        if (offset + ty >= r || target_col >= c)
            Nds[ty][tx] = 0;
        else 
            Nds[ty][tx] = B[(offset + ty) * r + target_col];
        
        // read-after-write dependence (True Dependence)
        // Threads must wait for data to be written to the proper place by other threads before reading it.
        // True dependence because thread truly needs the data supplied by writing thread, so it has no choice but to wait.
        __syncthreads(); 

        for(int k = 0; k < TILE_WIDTH; k++) {
            result += Mds[ty][k] * Nds[k][tx];
        }
        
        // write-after-read dependence (False Dependence)
        // Threads must wait for data to be read by all threads that need it before overwriting
        // False dependence because writing thread does not need any data from the reading thread.
        __syncthreads();
    }

    if (target_row < r && target_col < c)
        C[target_row * r + target_col] = result;
}

void matrix::cudaMultiplyV2(matrix *B, matrix *C) {
    if (this->cols != B->rows) {
        printf("Inconsistent shapes for matrix multiplication. matrixA.cols should be eequal to matrixB.rows");
        return;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, this->rows * this->cols * sizeof(int));
    cudaMalloc((void **)&d_B, B->rows * B->cols * sizeof(int));
    cudaMalloc((void **)&d_C, C->rows * C->cols * sizeof(int));

    cudaMemcpy(d_A, this->data, this->rows * this->cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, B->rows * B->cols * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(this->rows / (BLOCK_SIZE * 1.0)), ceil(this->rows / (BLOCK_SIZE * 1.0)));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiplyV2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, C->rows, C->cols);

    cudaMemcpy(C->data, d_C, this->rows * this->cols * sizeof(int), cudaMemcpyDeviceToHost);
}