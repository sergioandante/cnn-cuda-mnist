#include "layers.hpp"
#include "kernels.cuh"
#include "common.cuh"
#include <random>
#include <vector>
#include <cmath>

void Param::alloc(int n) {
    size = n;
    data = dalloc(n);
    grad = dalloc(n);
    m = dalloc(n);
    v = dalloc(n);
    dzero(grad, n);
    dzero(m, n);
    dzero(v, n);
}

void Param::zero_grad() { dzero(grad, size); }

void init_he(Param& p, int fan_in, unsigned seed) {
    std::mt19937 rng(seed);
    float std_dev = std::sqrt(2.0f / (float)fan_in);
    std::normal_distribution<float> dist(0.0f, std_dev);
    std::vector<float> h(p.size);
    for (int i = 0; i < p.size; ++i) h[i] = dist(rng);
    CUDA_CHECK(cudaMemcpy(p.data, h.data(), p.size * sizeof(float), cudaMemcpyHostToDevice));
}

void init_zero(Param& p) { dzero(p.data, p.size); }

// ===========================================================================
// Conv2D
// ===========================================================================
Conv2D::Conv2D(int inC_, int outC_, int KH_, int KW_, int padH_, int padW_,
               int strideH_, int strideW_, int maxBatch_, int H_, int Wd_, unsigned seed)
    : inC(inC_), outC(outC_), KH(KH_), KW(KW_), padH(padH_), padW(padW_),
      strideH(strideH_), strideW(strideW_), maxBatch(maxBatch_), H(H_), W(Wd_) {
    outH = (H + 2 * padH - KH) / strideH + 1;
    outW = (W + 2 * padW - KW) / strideW + 1;

    weight.alloc(outC * inC * KH * KW);
    bias.alloc(outC);
    init_he(weight, inC * KH * KW, seed);
    init_zero(bias);

    colRows_ = inC * KH * KW;
    size_t maxColCols = (size_t)maxBatch * outH * outW;
    col_ = dalloc(colRows_ * maxColCols);
    dcol_ = dalloc(colRows_ * maxColCols);
}

Conv2D::~Conv2D() {
    cudaFree(col_);
    cudaFree(dcol_);
}

void Conv2D::forward(const float* x, float* y, int n) {
    int colCols = n * outH * outW;
    im2col(x, inC, n, H, W, KH, KW, padH, padW, strideH, strideW, outH, outW, col_);
    // y[outC, colCols] = weight[outC, colRows_] * col[colRows_, colCols]
    gemm(weight.data, false, col_, false, y, outC, colCols, colRows_, 1.0f, 0.0f);
    add_bias_rowwise(y, bias.data, outC, colCols);
}

void Conv2D::backward(const float* x, const float* dy, float* dx, int n) {
    int colCols = n * outH * outW;
    // dWeight[outC, colRows_] = dy[outC, colCols] * col[colRows_, colCols]^T
    gemm(dy, false, col_, true, weight.grad, outC, colRows_, colCols, 1.0f, 0.0f);
    bias_grad_rowwise(dy, bias.grad, outC, colCols);

    if (dx != nullptr) {
        // dCol[colRows_, colCols] = weight[outC, colRows_]^T * dy[outC, colCols]
        gemm(weight.data, true, dy, false, dcol_, colRows_, colCols, outC, 1.0f, 0.0f);
        col2im(dcol_, dx, inC, n, H, W, KH, KW, padH, padW, strideH, strideW, outH, outW);
    }
}

std::vector<Param*> Conv2D::params() { return {&weight, &bias}; }

// ===========================================================================
// MaxPool2D
// ===========================================================================
MaxPool2D::MaxPool2D(int C_, int maxBatch_, int H_, int Wd_, int poolK_, int poolS_)
    : C(C_), maxBatch(maxBatch_), H(H_), W(Wd_), poolK(poolK_), poolS(poolS_) {
    outH = H / poolS;
    outW = W / poolS;
    CUDA_CHECK(cudaMalloc(&argmax_, sizeof(int) * (size_t)C * maxBatch * outH * outW));
}
MaxPool2D::~MaxPool2D() { cudaFree(argmax_); }

void MaxPool2D::forward(const float* x, float* y, int n) {
    maxpool_forward(x, y, argmax_, C, n, H, W, poolK, poolS, outH, outW);
}
void MaxPool2D::backward(const float* dy, float* dx, int n) {
    maxpool_backward(dy, argmax_, dx, C, n, H, W, outH, outW);
}

// ===========================================================================
// Linear
// ===========================================================================
Linear::Linear(int inF_, int outF_, int maxBatch_, unsigned seed)
    : inF(inF_), outF(outF_), maxBatch(maxBatch_) {
    weight.alloc(outF * inF);
    bias.alloc(outF);
    init_he(weight, inF, seed);
    init_zero(bias);
}

void Linear::forward(const float* x, float* y, int n) {
    // y[outF, n] = weight[outF, inF] * x[inF, n]
    gemm(weight.data, false, x, false, y, outF, n, inF, 1.0f, 0.0f);
    add_bias_rowwise(y, bias.data, outF, n);
}

void Linear::backward(const float* x, const float* dy, float* dx, int n) {
    // dWeight[outF, inF] = dy[outF, n] * x[inF, n]^T
    gemm(dy, false, x, true, weight.grad, outF, inF, n, 1.0f, 0.0f);
    bias_grad_rowwise(dy, bias.grad, outF, n);
    if (dx != nullptr) {
        // dx[inF, n] = weight[outF, inF]^T * dy[outF, n]
        gemm(weight.data, true, dy, false, dx, inF, n, outF, 1.0f, 0.0f);
    }
}

std::vector<Param*> Linear::params() { return {&weight, &bias}; }

// ===========================================================================
// Dropout
// ===========================================================================
Dropout::Dropout(int maxN_, float p_, curandGenerator_t gen) : maxN(maxN_), p(p_), gen_(gen) {
    randbuf_ = dalloc(maxN);
}
Dropout::~Dropout() { cudaFree(randbuf_); }

void Dropout::forward(const float* x, float* y, int n, bool training) {
    if (!training) {
        CUDA_CHECK(cudaMemcpy(y, x, sizeof(float) * n, cudaMemcpyDeviceToDevice));
        return;
    }
    CURAND_CHECK(curandGenerateUniform(gen_, randbuf_, n));
    dropout_forward(x, randbuf_, y, n, p);
}

void Dropout::backward(const float* dy, float* dx, int n) {
    dropout_backward(dy, randbuf_, dx, n, p);
}
