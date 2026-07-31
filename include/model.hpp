#pragma once
#include "layers.hpp"
#include <curand.h>
#include <vector>

// CNN estilo LeNet para MNIST: Conv(1->16,3x3)-ReLU-Pool -> Conv(16->32,3x3)-ReLU-Pool
// -> FC(1568->128)-ReLU-Dropout -> FC(128->10)-Softmax+CE.
// Todas las activaciones intermedias se reservan para un tamaño de lote
// máximo (maxBatch); cada llamada puede usar n <= maxBatch.
class CnnModel {
private:
    // Declarado primero para que se construya antes que 'dropout' (el orden
    // de inicialización de miembros en C++ sigue el orden de declaración).
    curandGenerator_t gen_;

public:
    CnnModel(int maxBatch, unsigned seed);
    ~CnnModel();

    void setInput(const float* hostImages, int n);
    void setLabels(const int* hostLabels, int n);

    void forward(int n, bool training);
    void backward(int n);
    void zeroGrad();
    void adamStep(float lr, int t);

    long long paramCount() const;
    std::vector<Param*> allParams();

    float* probsDevice() { return probs_; }
    float* lossDevice() { return lossPerSample_; }
    int* labelsDevice() { return labels_; }
    // Activaciones intermedias (post-ReLU, pre-pooling) para visualización cualitativa.
    float* conv1ActDevice() { return c1_; }
    float* conv2ActDevice() { return c2_; }

    int maxBatch;
    Conv2D conv1, conv2;
    MaxPool2D pool1, pool2;
    Linear fc1, fc2;
    Dropout dropout;

private:
    float* x_;
    float* c1_;
    float* p1_;
    float* c2_;
    float* p2_;
    float* flat_;
    float* f1_;
    float* d1_;
    float* logits_;
    float* probs_;
    float* lossPerSample_;
    int* labels_;

    float* dlogits_;
    float* dd1_;
    float* ddf1_;
    float* dflat_;
    float* dp2_;
    float* dc2_;
    float* dp1_;
    float* dc1_;

    float adamBeta1_ = 0.9f, adamBeta2_ = 0.999f, adamEps_ = 1e-8f, weightDecay_ = 1e-4f;
};
