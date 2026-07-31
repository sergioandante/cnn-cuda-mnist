#pragma once
#include <vector>
#include <curand.h>

// Un tensor de parámetros entrenables con sus buffers de gradiente y estado
// de momento/velocidad para AdamW.
struct Param {
    float* data = nullptr;
    float* grad = nullptr;
    float* m = nullptr;
    float* v = nullptr;
    int size = 0;

    void alloc(int n);
    void zero_grad();
};

// Inicialización He (Kaiming) para pesos que preceden a una activación ReLU.
void init_he(Param& p, int fan_in, unsigned seed);
void init_zero(Param& p);

// ---------------------------------------------------------------------------
// Capa convolucional 2D. Entrada/salida en layout CNHW (canal-mayor):
// índice = ((c*n + b)*H + h)*W + w, con b < n <= maxBatch.
// Implementada vía im2col + GEMM. Los buffers internos se reservan para
// maxBatch, pero cada llamada a forward/backward puede usar un tamaño de
// lote real n <= maxBatch (para el último lote incompleto de una época).
// ---------------------------------------------------------------------------
class Conv2D {
public:
    Conv2D(int inC, int outC, int KH, int KW, int padH, int padW, int strideH,
           int strideW, int maxBatch, int H, int Wd, unsigned seed);
    ~Conv2D();

    void forward(const float* x, float* y, int n);
    // dx puede ser nullptr si no se necesita el gradiente de la entrada.
    void backward(const float* x, const float* dy, float* dx, int n);
    std::vector<Param*> params();

    int inC, outC, KH, KW, padH, padW, strideH, strideW, maxBatch, H, W;
    int outH, outW;
    Param weight;  // [outC, inC*KH*KW]
    Param bias;    // [outC]

private:
    float* col_;
    float* dcol_;
    int colRows_;
};

// ---------------------------------------------------------------------------
// Max-pooling 2D con ventanas disjuntas (poolK == poolS).
// ---------------------------------------------------------------------------
class MaxPool2D {
public:
    MaxPool2D(int C, int maxBatch, int H, int Wd, int poolK, int poolS);
    ~MaxPool2D();
    void forward(const float* x, float* y, int n);
    void backward(const float* dy, float* dx, int n);
    int C, maxBatch, H, W, poolK, poolS, outH, outW;

private:
    int* argmax_;
};

// ---------------------------------------------------------------------------
// Capa totalmente conectada. x, y en layout feature-mayor: [features, n].
// ---------------------------------------------------------------------------
class Linear {
public:
    Linear(int inF, int outF, int maxBatch, unsigned seed);
    void forward(const float* x, float* y, int n);
    void backward(const float* x, const float* dy, float* dx, int n);
    std::vector<Param*> params();

    int inF, outF, maxBatch;
    Param weight;  // [outF, inF]
    Param bias;    // [outF]
};

// ---------------------------------------------------------------------------
// Dropout con máscara generada por cuRAND (API de host).
// ---------------------------------------------------------------------------
class Dropout {
public:
    Dropout(int maxN, float p, curandGenerator_t gen);
    ~Dropout();
    void forward(const float* x, float* y, int n, bool training);
    void backward(const float* dy, float* dx, int n);
    int maxN;
    float p;

private:
    float* randbuf_;
    curandGenerator_t gen_;
};
