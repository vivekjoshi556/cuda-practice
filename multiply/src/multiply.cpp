#include <thread>
#include <stdio.h>
#include "multiply.h"

matrix::matrix(int row, int col) {
    this->rows = row;
    this->cols = col;
    this->data = new int[row * col](); // parans make sure that all elements are zero.
}

matrix::~matrix() {
    delete[] this->data;
}

void matrix::init() {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) 
            scanf("%d", &this->data[i * this->cols + j]);
    }
}

void matrix::show() {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++)
            printf("%d ", this->data[i * this->cols + j]);
        printf("\n");
    }
}

bool matrix::matches(matrix *A) {
    if(this->cols != A->cols || this->rows != A->rows) {
        printf("Match failed because of different Shapes\n");
        return false;
    }

    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            int idx = i * this->cols + j;
            if (this->data[idx] != A->data[idx]) {
                printf("Match failed because of different values at: (%d, %d). Expected %d and Got %d\n", i, j, A->data[idx], this->data[idx]);
                return false;
            }
        }
    }
    
    return true;
}

void matrix::clone(matrix *A) {
    // assert(A->rows == this->rows && A->cols == this->cols);
    
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            this->data[i * this->cols + j] = A->data[i * this->cols + j];
        }
    }
}

void matrix::multiplySerial(matrix *B, matrix *C) {
    if (this->cols != B->rows) {
        printf("Inconsistent shapes for matrix multiplication. matrixA.cols should be eequal to matrixB.rows");
        return;
    }

    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < B->cols; j++) {
            int total = 0;
            int idxA = i * B->cols;
            int idxB = j;
            for(int k = 0; k < this->cols; k++, idxA++, idxB += B->cols) {
                total += this->data[idxA] * B->data[idxB];
            }

            C->data[i * B->cols + j] = total;
        }
    }

}

void multiplicationThread(matrixThreadData *data) {
    for(int i = data->threadId; i < data->A->rows; i += data->numThreads) {
        for(int j = 0; j < data->B->cols; j++) {
            int total = 0;
            int idxA = i * data->B->cols;
            int idxB = j;
            for(int k = 0; k < data->A->cols; k++, idxA++, idxB += data->B->cols) {
                total += data->A->data[idxA] * data->B->data[idxB];
            }

            data->C->data[i * data->B->cols + j] = total;
        }
    }
}

void matrix::threadedMultiply(matrix *B, matrix *C, int numThreads) {
    if (this->cols != B->rows) {
        printf("Inconsistent shapes for matrix multiplication. matrixA.cols should be eequal to matrixB.rows");
        return;
    }

    matrixThreadData data[numThreads];
    std::thread workers[numThreads];

    for(int i = 0; i < numThreads; i++) {
        data[i].threadId = i;
        data[i].numThreads = numThreads;
        data[i].A = this;
        data[i].B = B;
        data[i].C = C;
        
        if(i < numThreads - 1)
            workers[i] = std::thread(multiplicationThread, &data[i]);
    }

    multiplicationThread(&data[3]);

    for(int i = 0; i < numThreads - 1; i++) {
        workers[i].join();
    }
}