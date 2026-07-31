#include "model.hpp"
#include "kernels.cuh"
#include "common.cuh"

static curandGenerator_t make_generator(unsigned seed) {
    curandGenerator_t g;
    CURAND_CHECK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(g, seed));
    return g;
}

CnnModel::CnnModel(int maxBatch_, unsigned seed)
    : gen_(make_generator(seed + 100)),
      maxBatch(maxBatch_),
      conv1(1, 16, 3, 3, 1, 1, 1, 1, maxBatch_, 28, 28, seed + 1),
      conv2(16, 32, 3, 3, 1, 1, 1, 1, maxBatch_, 14, 14, seed + 2),
      pool1(16, maxBatch_, 28, 28, 2, 2),
      pool2(32, maxBatch_, 14, 14, 2, 2),
      fc1(32 * 7 * 7, 128, maxBatch_, seed + 3),
      fc2(128, 10, maxBatch_, seed + 4),
      dropout(128 * maxBatch_, 0.25f, gen_) {
    x_ = dalloc((size_t)1 * maxBatch * 28 * 28);
    c1_ = dalloc((size_t)16 * maxBatch * 28 * 28);
    p1_ = dalloc((size_t)16 * maxBatch * 14 * 14);
    c2_ = dalloc((size_t)32 * maxBatch * 14 * 14);
    p2_ = dalloc((size_t)32 * maxBatch * 7 * 7);
    flat_ = dalloc((size_t)32 * 7 * 7 * maxBatch);
    f1_ = dalloc((size_t)128 * maxBatch);
    d1_ = dalloc((size_t)128 * maxBatch);
    logits_ = dalloc((size_t)10 * maxBatch);
    probs_ = dalloc((size_t)10 * maxBatch);
    lossPerSample_ = dalloc((size_t)maxBatch);
    CUDA_CHECK(cudaMalloc(&labels_, sizeof(int) * maxBatch));

    dlogits_ = dalloc((size_t)10 * maxBatch);
    dd1_ = dalloc((size_t)128 * maxBatch);
    ddf1_ = dalloc((size_t)128 * maxBatch);
    dflat_ = dalloc((size_t)32 * 7 * 7 * maxBatch);
    dp2_ = dalloc((size_t)32 * maxBatch * 7 * 7);
    dc2_ = dalloc((size_t)32 * maxBatch * 14 * 14);
    dp1_ = dalloc((size_t)16 * maxBatch * 14 * 14);
    dc1_ = dalloc((size_t)16 * maxBatch * 28 * 28);
}

CnnModel::~CnnModel() {
    cudaFree(x_); cudaFree(c1_); cudaFree(p1_); cudaFree(c2_); cudaFree(p2_);
    cudaFree(flat_); cudaFree(f1_); cudaFree(d1_); cudaFree(logits_); cudaFree(probs_);
    cudaFree(lossPerSample_); cudaFree(labels_);
    cudaFree(dlogits_); cudaFree(dd1_); cudaFree(ddf1_); cudaFree(dflat_);
    cudaFree(dp2_); cudaFree(dc2_); cudaFree(dp1_); cudaFree(dc1_);
    curandDestroyGenerator(gen_);
}

void CnnModel::setInput(const float* hostImages, int n) {
    CUDA_CHECK(cudaMemcpy(x_, hostImages, sizeof(float) * (size_t)1 * n * 28 * 28,
                           cudaMemcpyHostToDevice));
}

void CnnModel::setLabels(const int* hostLabels, int n) {
    CUDA_CHECK(cudaMemcpy(labels_, hostLabels, sizeof(int) * n, cudaMemcpyHostToDevice));
}

void CnnModel::forward(int n, bool training) {
    conv1.forward(x_, c1_, n);
    relu_forward(c1_, c1_, 16 * n * 28 * 28);
    pool1.forward(c1_, p1_, n);

    conv2.forward(p1_, c2_, n);
    relu_forward(c2_, c2_, 32 * n * 14 * 14);
    pool2.forward(c2_, p2_, n);

    flatten_chw_to_fn(p2_, flat_, 32, n, 7, 7);

    fc1.forward(flat_, f1_, n);
    relu_forward(f1_, f1_, 128 * n);
    dropout.forward(f1_, d1_, 128 * n, training);

    fc2.forward(d1_, logits_, n);
    softmax_xent_forward(logits_, labels_, lossPerSample_, probs_, 10, n);
}

void CnnModel::backward(int n) {
    softmax_xent_backward(probs_, labels_, dlogits_, 10, n);
    fc2.backward(d1_, dlogits_, dd1_, n);
    dropout.backward(dd1_, ddf1_, 128 * n);
    relu_backward(f1_, ddf1_, ddf1_, 128 * n);
    fc1.backward(flat_, ddf1_, dflat_, n);
    unflatten_fn_to_chw(dflat_, dp2_, 32, n, 7, 7);
    pool2.backward(dp2_, dc2_, n);
    relu_backward(c2_, dc2_, dc2_, 32 * n * 14 * 14);
    conv2.backward(p1_, dc2_, dp1_, n);
    pool1.backward(dp1_, dc1_, n);
    relu_backward(c1_, dc1_, dc1_, 16 * n * 28 * 28);
    conv1.backward(x_, dc1_, nullptr, n);
}

void CnnModel::zeroGrad() {
    for (Param* p : allParams()) p->zero_grad();
}

void CnnModel::adamStep(float lr, int t) {
    for (Param* p : allParams()) {
        adamw_step(p->data, p->grad, p->m, p->v, p->size, lr, adamBeta1_, adamBeta2_,
                   adamEps_, weightDecay_, t);
    }
}

long long CnnModel::paramCount() const {
    long long total = 0;
    for (Param* p : const_cast<CnnModel*>(this)->allParams()) total += p->size;
    return total;
}

std::vector<Param*> CnnModel::allParams() {
    std::vector<Param*> all;
    for (auto* p : conv1.params()) all.push_back(p);
    for (auto* p : conv2.params()) all.push_back(p);
    for (auto* p : fc1.params()) all.push_back(p);
    for (auto* p : fc2.params()) all.push_back(p);
    return all;
}
