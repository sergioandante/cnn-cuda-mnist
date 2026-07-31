#pragma once
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// GEMM genérico: C[M x N] = alpha * op(A) * op(B) + beta * C
// A es [M x K] o [K x M] si transA, B es [K x N] o [N x K] si transB.
// Todas las matrices están en formato row-major.
// ---------------------------------------------------------------------------
void gemm(const float* A, bool transA, const float* B, bool transB, float* C,
          int M, int N, int K, float alpha, float beta);

// im2col / col2im para convolución 2D (entrada en layout CNHW: canal-mayor,
// es decir índice = ((c*Nb + n)*H + h)*W + w).
// data_col resultante tiene forma [(C*KH*KW) x (Nb*outH*outW)].
void im2col(const float* data_im, int C, int Nb, int H, int W, int KH, int KW,
            int padH, int padW, int strideH, int strideW, int outH, int outW,
            float* data_col);

void col2im(const float* data_col, float* data_im, int C, int Nb, int H, int W,
            int KH, int KW, int padH, int padW, int strideH, int strideW,
            int outH, int outW);

// Suma un sesgo por fila: out[r, c] += bias[r], out tiene forma [rows x cols]
void add_bias_rowwise(float* out, const float* bias, int rows, int cols);

// Gradiente del sesgo: dBias[r] = sum_c dOut[r, c]
void bias_grad_rowwise(const float* dOut, float* dBias, int rows, int cols);

// Activación ReLU
void relu_forward(const float* x, float* y, int n);
void relu_backward(const float* x, const float* dy, float* dx, int n);

// Max-pooling 2D (layout CNHW), ventana poolK x poolK, stride poolS, sin solape.
void maxpool_forward(const float* x, float* y, int* argmax, int C, int Nb, int H,
                      int W, int poolK, int poolS, int outH, int outW);
void maxpool_backward(const float* dy, const int* argmax, float* dx, int C,
                       int Nb, int H, int W, int outH, int outW);

// Aplana activaciones CNHW [C,Nb,H,W] a formato feature-mayor [C*H*W, Nb] y viceversa.
void flatten_chw_to_fn(const float* x, float* y, int C, int Nb, int H, int W);
void unflatten_fn_to_chw(const float* dy, float* dx, int C, int Nb, int H, int W);

// Dropout: rand ~ U(0,1) ya generado externamente (cuRAND) en 'randbuf'.
void dropout_forward(const float* x, const float* randbuf, float* y, int n, float p);
void dropout_backward(const float* dy, const float* randbuf, float* dx, int n, float p);

// Softmax + entropía cruzada combinados. logits/probs en formato [C x N] (clase-mayor).
void softmax_xent_forward(const float* logits, const int* labels, float* loss_per_sample,
                           float* probs, int C, int N);
void softmax_xent_backward(const float* probs, const int* labels, float* dlogits, int C, int N);

// Predicción (argmax por columna) sobre probs [C x N] -> preds[N]
void argmax_cols(const float* probs, int* preds, int C, int N);

// Paso de optimización AdamW para un tensor de parámetros de tamaño n.
void adamw_step(float* param, const float* grad, float* m, float* v, int n,
                 float lr, float beta1, float beta2, float eps, float weight_decay, int t);
