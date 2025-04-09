#ifndef MULTIPLY_HPP
#define MULTIPLY_HPP

#define TILE_WIDTH 32
#define BLOCK_SIZE 32

class matrix {
public:
    int rows, cols;
    int *data = nullptr;

    matrix(int, int);
    ~matrix();
    void init();

    void show();

    void clone(matrix*);
    
    bool matches(matrix*);

    void threadedMultiply(matrix*, matrix*, int);
    
    void multiplySerial(matrix*, matrix*);

    void cudaMultiplyV1(matrix*, matrix*);

    void cudaMultiplyV2(matrix*, matrix*);
};

class matrixThreadData {
public:
    int rowId;
    int colId;
    int threadId;
    int numThreads;
    matrix *A, *B, *C;
};

void multiplicationThread(matrixThreadData*);

void getDeviceSpecs();

#endif