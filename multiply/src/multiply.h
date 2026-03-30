#ifndef MULTIPLY_HPP
#define MULTIPLY_HPP

#define TILE_WIDTH 32
#define BLOCK_SIZE 32
#define FULL_MASK 0xffffffff

class matrix {
public:
    int rows, cols;
    float *data = nullptr;

    matrix(int, int);
    ~matrix();
    void init(bool);

    void show();

    void clone(matrix*);
    
    bool matches(matrix*);

    void clear();

    void threadedMultiply(matrix*, matrix*, int);
    
    void multiplySerial(matrix*, matrix*);

    void cudaMultiply(matrix*, matrix*, int);
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