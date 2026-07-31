#include "kernels.cuh"
#include "common.cuh"
#include <cmath>

// ===========================================================================
// GEMM tileado con memoria compartida, con soporte de transposición lógica.
// ===========================================================================
#define TILE 16

template <bool TA, bool TB>
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K,
                             float alpha, float beta) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = TA ? A[aCol * M + row] : A[row * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * TILE + threadIdx.y;
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = TB ? B[col * K + bRow] : B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float c_old = (beta != 0.0f) ? C[row * N + col] : 0.0f;
        C[row * N + col] = alpha * acc + beta * c_old;
    }
}

void gemm(const float* A, bool transA, const float* B, bool transB, float* C,
          int M, int N, int K, float alpha, float beta) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    if (!transA && !transB) gemm_kernel<false, false><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    else if (transA && !transB) gemm_kernel<true, false><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    else if (!transA && transB) gemm_kernel<false, true><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    else gemm_kernel<true, true><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// im2col / col2im (layout de entrada CNHW: ((c*Nb+n)*H+h)*W+w)
// ===========================================================================
__global__ void im2col_kernel(const float* data_im, float* data_col, int C, int Nb,
                               int H, int W, int KH, int KW, int padH, int padW,
                               int strideH, int strideW, int outH, int outW) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = Nb * outH * outW;
    if (col_idx >= total_cols) return;

    int ow = col_idx % outW;
    int oh = (col_idx / outW) % outH;
    int n = col_idx / (outW * outH);

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in = oh * strideH - padH + kh;
                int w_in = ow * strideW - padW + kw;
                int row = (c * KH + kh) * KW + kw;
                float val = 0.0f;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    val = data_im[((c * Nb + n) * H + h_in) * W + w_in];
                }
                data_col[row * total_cols + col_idx] = val;
            }
        }
    }
}

void im2col(const float* data_im, int C, int Nb, int H, int W, int KH, int KW,
            int padH, int padW, int strideH, int strideW, int outH, int outW,
            float* data_col) {
    int total_cols = Nb * outH * outW;
    int threads = 256;
    int blocks = (total_cols + threads - 1) / threads;
    im2col_kernel<<<blocks, threads>>>(data_im, data_col, C, Nb, H, W, KH, KW, padH,
                                        padW, strideH, strideW, outH, outW);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void col2im_kernel(const float* data_col, float* data_im, int C, int Nb,
                               int H, int W, int KH, int KW, int padH, int padW,
                               int strideH, int strideW, int outH, int outW) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = Nb * outH * outW;
    if (col_idx >= total_cols) return;

    int ow = col_idx % outW;
    int oh = (col_idx / outW) % outH;
    int n = col_idx / (outW * outH);

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in = oh * strideH - padH + kh;
                int w_in = ow * strideW - padW + kw;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int row = (c * KH + kh) * KW + kw;
                    float g = data_col[row * total_cols + col_idx];
                    atomicAdd(&data_im[((c * Nb + n) * H + h_in) * W + w_in], g);
                }
            }
        }
    }
}

void col2im(const float* data_col, float* data_im, int C, int Nb, int H, int W,
            int KH, int KW, int padH, int padW, int strideH, int strideW,
            int outH, int outW) {
    CUDA_CHECK(cudaMemset(data_im, 0, sizeof(float) * (size_t)C * Nb * H * W));
    int total_cols = Nb * outH * outW;
    int threads = 256;
    int blocks = (total_cols + threads - 1) / threads;
    col2im_kernel<<<blocks, threads>>>(data_col, data_im, C, Nb, H, W, KH, KW, padH,
                                        padW, strideH, strideW, outH, outW);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Sesgo por fila (compartido entre capas convolucionales y lineales)
// ===========================================================================
__global__ void add_bias_rowwise_kernel(float* out, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    out[idx] += bias[r];
}

void add_bias_rowwise(float* out, const float* bias, int rows, int cols) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_rowwise_kernel<<<blocks, threads>>>(out, bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void bias_grad_rowwise_kernel(const float* dOut, float* dBias, int cols) {
    extern __shared__ float sdata[];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) sum += dOut[r * cols + c];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) dBias[r] = sdata[0];
}

void bias_grad_rowwise(const float* dOut, float* dBias, int rows, int cols) {
    int threads = 256;
    bias_grad_rowwise_kernel<<<rows, threads, threads * sizeof(float)>>>(dOut, dBias, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// ReLU
// ===========================================================================
__global__ void relu_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
__global__ void relu_backward_kernel(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}
void relu_forward(const float* x, float* y, int n) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(x, y, n);
    CUDA_CHECK(cudaGetLastError());
}
void relu_backward(const float* x, const float* dy, float* dx, int n) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(x, dy, dx, n);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Max-pooling 2D (ventanas disjuntas, layout CNHW)
// ===========================================================================
__global__ void maxpool_forward_kernel(const float* x, float* y, int* argmax, int C,
                                        int Nb, int H, int W, int poolK, int poolS,
                                        int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * Nb * outH * outW;
    if (idx >= total) return;
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int n = (idx / (outW * outH)) % Nb;
    int c = idx / (outW * outH * Nb);
    int h0 = oh * poolS, w0 = ow * poolS;
    float best = -3.0e38f;
    int bestIdx = -1;
    for (int kh = 0; kh < poolK; ++kh) {
        for (int kw = 0; kw < poolK; ++kw) {
            int h = h0 + kh, w = w0 + kw;
            int in_idx = ((c * Nb + n) * H + h) * W + w;
            float v = x[in_idx];
            if (v > best) { best = v; bestIdx = in_idx; }
        }
    }
    y[idx] = best;
    argmax[idx] = bestIdx;
}
void maxpool_forward(const float* x, float* y, int* argmax, int C, int Nb, int H,
                      int W, int poolK, int poolS, int outH, int outW) {
    int total = C * Nb * outH * outW;
    int threads = 256, blocks = (total + threads - 1) / threads;
    maxpool_forward_kernel<<<blocks, threads>>>(x, y, argmax, C, Nb, H, W, poolK, poolS, outH, outW);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool_backward_kernel(const float* dy, const int* argmax, float* dx, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dx[argmax[idx]] = dy[idx];
}
void maxpool_backward(const float* dy, const int* argmax, float* dx, int C, int Nb,
                       int H, int W, int outH, int outW) {
    CUDA_CHECK(cudaMemset(dx, 0, sizeof(float) * (size_t)C * Nb * H * W));
    int total = C * Nb * outH * outW;
    int threads = 256, blocks = (total + threads - 1) / threads;
    maxpool_backward_kernel<<<blocks, threads>>>(dy, argmax, dx, total);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Aplanado CNHW <-> feature-mayor [C*H*W, Nb]
// ===========================================================================
__global__ void flatten_kernel(const float* x, float* y, int C, int Nb, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * Nb * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int n = (idx / (W * H)) % Nb;
    int c = idx / (W * H * Nb);
    int f = (c * H + h) * W + w;
    y[f * Nb + n] = x[idx];
}
void flatten_chw_to_fn(const float* x, float* y, int C, int Nb, int H, int W) {
    int total = C * Nb * H * W;
    int threads = 256, blocks = (total + threads - 1) / threads;
    flatten_kernel<<<blocks, threads>>>(x, y, C, Nb, H, W);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void unflatten_kernel(const float* dy, float* dx, int C, int Nb, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * Nb * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int n = (idx / (W * H)) % Nb;
    int c = idx / (W * H * Nb);
    int f = (c * H + h) * W + w;
    dx[idx] = dy[f * Nb + n];
}
void unflatten_fn_to_chw(const float* dy, float* dx, int C, int Nb, int H, int W) {
    int total = C * Nb * H * W;
    int threads = 256, blocks = (total + threads - 1) / threads;
    unflatten_kernel<<<blocks, threads>>>(dy, dx, C, Nb, H, W);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Dropout (la máscara aleatoria U(0,1) se genera externamente con cuRAND)
// ===========================================================================
__global__ void dropout_forward_kernel(const float* x, const float* r, float* y, int n, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (r[i] > p) ? x[i] / (1.0f - p) : 0.0f;
}
__global__ void dropout_backward_kernel(const float* dy, const float* r, float* dx, int n, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = (r[i] > p) ? dy[i] / (1.0f - p) : 0.0f;
}
void dropout_forward(const float* x, const float* randbuf, float* y, int n, float p) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    dropout_forward_kernel<<<blocks, threads>>>(x, randbuf, y, n, p);
    CUDA_CHECK(cudaGetLastError());
}
void dropout_backward(const float* dy, const float* randbuf, float* dx, int n, float p) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    dropout_backward_kernel<<<blocks, threads>>>(dy, randbuf, dx, n, p);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Softmax + entropía cruzada (logits/probs en formato [C x N], clase-mayor)
// ===========================================================================
__global__ void softmax_xent_forward_kernel(const float* logits, const int* labels,
                                             float* loss, float* probs, int C, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float maxv = -3.0e38f;
    for (int c = 0; c < C; ++c) {
        float v = logits[c * N + n];
        if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float e = expf(logits[c * N + n] - maxv);
        probs[c * N + n] = e;
        sum += e;
    }
    for (int c = 0; c < C; ++c) probs[c * N + n] /= sum;
    int label = labels[n];
    float p = probs[label * N + n];
    loss[n] = -logf(fmaxf(p, 1e-12f));
}
void softmax_xent_forward(const float* logits, const int* labels, float* loss_per_sample,
                           float* probs, int C, int N) {
    int threads = 128, blocks = (N + threads - 1) / threads;
    softmax_xent_forward_kernel<<<blocks, threads>>>(logits, labels, loss_per_sample, probs, C, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void softmax_xent_backward_kernel(const float* probs, const int* labels,
                                              float* dlogits, int C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * N;
    if (idx >= total) return;
    int n = idx % N;
    int c = idx / N;
    float y = (labels[n] == c) ? 1.0f : 0.0f;
    dlogits[idx] = (probs[idx] - y) / (float)N;
}
void softmax_xent_backward(const float* probs, const int* labels, float* dlogits, int C, int N) {
    int total = C * N;
    int threads = 256, blocks = (total + threads - 1) / threads;
    softmax_xent_backward_kernel<<<blocks, threads>>>(probs, labels, dlogits, C, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void argmax_cols_kernel(const float* probs, int* preds, int C, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float best = -1.0f;
    int bi = 0;
    for (int c = 0; c < C; ++c) {
        float v = probs[c * N + n];
        if (v > best) { best = v; bi = c; }
    }
    preds[n] = bi;
}
void argmax_cols(const float* probs, int* preds, int C, int N) {
    int threads = 128, blocks = (N + threads - 1) / threads;
    argmax_cols_kernel<<<blocks, threads>>>(probs, preds, C, N);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// AdamW
// ===========================================================================
__global__ void adamw_step_kernel(float* param, const float* grad, float* m, float* v,
                                   int n, float lr, float beta1, float beta2, float eps,
                                   float weight_decay, float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    param[i] -= lr * (mhat / (sqrtf(vhat) + eps) + weight_decay * param[i]);
}
void adamw_step(float* param, const float* grad, float* m, float* v, int n, float lr,
                 float beta1, float beta2, float eps, float weight_decay, int t) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    int threads = 256, blocks = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads>>>(param, grad, m, v, n, lr, beta1, beta2, eps,
                                            weight_decay, bc1, bc2);
    CUDA_CHECK(cudaGetLastError());
}
