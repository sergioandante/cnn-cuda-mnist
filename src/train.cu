#include "mnist.hpp"
#include "model.hpp"
#include "kernels.cuh"
#include "common.cuh"

#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Hiperparámetros
// ---------------------------------------------------------------------------
static const int kMaxBatch = 128;
static const int kEpochs = 20;
static const float kLrMax = 1.0e-3f;
static const float kLrMin = 1.0e-6f;
static const float kWarmupFraction = 0.10f;
static const float kPi = 3.14159265358979323846f;

static float lr_schedule(int step, int warmupSteps, int totalSteps, float lrMax, float lrMin) {
    if (step <= warmupSteps) {
        return lrMax * (float)step / (float)std::max(1, warmupSteps);
    }
    float progress = (float)(step - warmupSteps) / (float)std::max(1, totalSteps - warmupSteps);
    progress = std::min(1.0f, progress);
    return lrMin + 0.5f * (lrMax - lrMin) * (1.0f + std::cos(kPi * progress));
}

// Ejecuta el modelo sobre un conjunto completo (train o test) en modo
// evaluación o entrenamiento, en lotes de tamaño <= kMaxBatch, y opcionalmente
// realiza el paso de retropropagación + optimización.
struct EpochStats {
    double avgLoss = 0.0;
    double accuracy = 0.0;
};

static EpochStats run_epoch(CnnModel& model, const MnistDataset& ds,
                             const std::vector<int>& order, bool training,
                             int* globalStep, int warmupSteps, int totalSteps,
                             std::vector<float>& batchImgBuf, std::vector<int>& batchLblBuf,
                             std::vector<float>& lossHostBuf, std::vector<int>& predHostBuf,
                             int* argmaxDevice) {
    const int imgSize = ds.rows * ds.cols;
    int numSamples = (int)order.size();
    int numBatches = (numSamples + kMaxBatch - 1) / kMaxBatch;

    double totalLoss = 0.0;
    long long totalCorrect = 0;

    for (int b = 0; b < numBatches; ++b) {
        int start = b * kMaxBatch;
        int n = std::min(kMaxBatch, numSamples - start);

        for (int i = 0; i < n; ++i) {
            int idx = order[start + i];
            std::copy(ds.images.begin() + (size_t)idx * imgSize,
                      ds.images.begin() + (size_t)(idx + 1) * imgSize,
                      batchImgBuf.begin() + (size_t)i * imgSize);
            batchLblBuf[i] = ds.labels[idx];
        }

        model.setInput(batchImgBuf.data(), n);
        model.setLabels(batchLblBuf.data(), n);

        if (training) {
            float lr = lr_schedule(*globalStep, warmupSteps, totalSteps, kLrMax, kLrMin);
            model.forward(n, /*training=*/true);
            model.backward(n);
            model.adamStep(lr, *globalStep);
            (*globalStep)++;
        } else {
            model.forward(n, /*training=*/false);
        }

        CUDA_CHECK(cudaMemcpy(lossHostBuf.data(), model.lossDevice(), sizeof(float) * n,
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) totalLoss += lossHostBuf[i];

        argmax_cols(model.probsDevice(), argmaxDevice, 10, n);
        CUDA_CHECK(cudaMemcpy(predHostBuf.data(), argmaxDevice, sizeof(int) * n,
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            if (predHostBuf[i] == batchLblBuf[i]) totalCorrect++;
        }
    }

    EpochStats st;
    st.avgLoss = totalLoss / (double)numSamples;
    st.accuracy = (double)totalCorrect / (double)numSamples;
    return st;
}

int main() {
    std::cout << "Cargando MNIST...\n";
    MnistDataset train = load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MnistDataset test = load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    std::cout << "Entrenamiento: " << train.numImages << " imagenes, Prueba: "
              << test.numImages << " imagenes (" << train.rows << "x" << train.cols << ")\n";

    unsigned seed = 42;
    CnnModel model(kMaxBatch, seed);

    long long totalParams = model.paramCount();
    std::cout << "Parametros totales del modelo: " << totalParams << "\n";

    size_t freeB, totalB;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    double usedMiB = (double)(totalB - freeB) / (1024.0 * 1024.0);
    std::cout << "Memoria GPU en uso tras inicializar el modelo: " << usedMiB << " MiB\n";

    int batchesPerEpoch = (train.numImages + kMaxBatch - 1) / kMaxBatch;
    int totalSteps = batchesPerEpoch * kEpochs;
    int warmupSteps = std::max(1, (int)(kWarmupFraction * totalSteps));
    std::cout << "Lotes por epoca: " << batchesPerEpoch << ", pasos totales: " << totalSteps
              << ", pasos de warmup: " << warmupSteps << "\n";

    std::vector<int> trainOrder(train.numImages);
    std::iota(trainOrder.begin(), trainOrder.end(), 0);
    std::vector<int> testOrder(test.numImages);
    std::iota(testOrder.begin(), testOrder.end(), 0);
    std::mt19937 rng(123);

    std::vector<float> batchImgBuf((size_t)kMaxBatch * 28 * 28);
    std::vector<int> batchLblBuf(kMaxBatch);
    std::vector<float> lossHostBuf(kMaxBatch);
    std::vector<int> predHostBuf(kMaxBatch);
    int* argmaxDevice;
    CUDA_CHECK(cudaMalloc(&argmaxDevice, sizeof(int) * kMaxBatch));

    std::ofstream csv("results/training_log.csv");
    csv << "epoch,train_loss,train_acc,test_loss,test_acc,lr,epoch_time_s\n";

    int globalStep = 1;
    double bestTestAcc = 0.0;
    int bestEpoch = -1;

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        std::shuffle(trainOrder.begin(), trainOrder.end(), rng);
        auto t0 = std::chrono::high_resolution_clock::now();

        EpochStats trainStats = run_epoch(model, train, trainOrder, true, &globalStep,
                                           warmupSteps, totalSteps, batchImgBuf, batchLblBuf,
                                           lossHostBuf, predHostBuf, argmaxDevice);

        EpochStats testStats = run_epoch(model, test, testOrder, false, &globalStep,
                                          warmupSteps, totalSteps, batchImgBuf, batchLblBuf,
                                          lossHostBuf, predHostBuf, argmaxDevice);

        auto t1 = std::chrono::high_resolution_clock::now();
        double epochTime = std::chrono::duration<double>(t1 - t0).count();

        float lrAtEpochEnd = lr_schedule(globalStep - 1, warmupSteps, totalSteps, kLrMax, kLrMin);

        if (testStats.accuracy > bestTestAcc) {
            bestTestAcc = testStats.accuracy;
            bestEpoch = epoch;
        }

        std::cout << "Epoca " << epoch << "/" << kEpochs
                  << " | train_loss=" << trainStats.avgLoss
                  << " train_acc=" << trainStats.accuracy
                  << " | test_loss=" << testStats.avgLoss
                  << " test_acc=" << testStats.accuracy
                  << " | lr=" << lrAtEpochEnd
                  << " | tiempo=" << epochTime << "s\n";

        csv << epoch << "," << trainStats.avgLoss << "," << trainStats.accuracy << ","
            << testStats.avgLoss << "," << testStats.accuracy << "," << lrAtEpochEnd << ","
            << epochTime << "\n";
        csv.flush();
    }
    csv.close();

    std::cout << "Mejor exactitud de prueba: " << bestTestAcc << " en la epoca " << bestEpoch << "\n";

    // -----------------------------------------------------------------------
    // Matriz de confusion final sobre el conjunto de prueba completo (10000).
    // -----------------------------------------------------------------------
    std::vector<std::vector<long long>> confusion(10, std::vector<long long>(10, 0));
    {
        int numSamples = test.numImages;
        int numBatches = (numSamples + kMaxBatch - 1) / kMaxBatch;
        const int imgSize = test.rows * test.cols;
        for (int b = 0; b < numBatches; ++b) {
            int start = b * kMaxBatch;
            int n = std::min(kMaxBatch, numSamples - start);
            for (int i = 0; i < n; ++i) {
                int idx = testOrder[start + i];
                std::copy(test.images.begin() + (size_t)idx * imgSize,
                          test.images.begin() + (size_t)(idx + 1) * imgSize,
                          batchImgBuf.begin() + (size_t)i * imgSize);
                batchLblBuf[i] = test.labels[idx];
            }
            model.setInput(batchImgBuf.data(), n);
            model.setLabels(batchLblBuf.data(), n);
            model.forward(n, false);
            argmax_cols(model.probsDevice(), argmaxDevice, 10, n);
            CUDA_CHECK(cudaMemcpy(predHostBuf.data(), argmaxDevice, sizeof(int) * n,
                                   cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                confusion[batchLblBuf[i]][predHostBuf[i]]++;
            }
        }
    }
    {
        std::ofstream f("results/confusion_matrix.csv");
        for (int r = 0; r < 10; ++r) {
            for (int c = 0; c < 10; ++c) {
                f << confusion[r][c];
                if (c < 9) f << ",";
            }
            f << "\n";
        }
    }
    std::cout << "Matriz de confusion escrita en results/confusion_matrix.csv\n";

    // -----------------------------------------------------------------------
    // Exportar filtros de conv1 (16 filtros de 1x3x3).
    // -----------------------------------------------------------------------
    {
        std::vector<float> filt(model.conv1.weight.size);
        CUDA_CHECK(cudaMemcpy(filt.data(), model.conv1.weight.data,
                               sizeof(float) * filt.size(), cudaMemcpyDeviceToHost));
        std::ofstream f("results/conv1_filters.csv");
        for (size_t i = 0; i < filt.size(); ++i) {
            f << filt[i];
            if (i + 1 < filt.size()) f << ",";
        }
        f << "\n";
    }

    // -----------------------------------------------------------------------
    // Mapas de activacion de conv1 y conv2 para una imagen de muestra (n=1).
    // -----------------------------------------------------------------------
    {
        int sampleIdx = testOrder[7];
        const int imgSize = test.rows * test.cols;
        std::copy(test.images.begin() + (size_t)sampleIdx * imgSize,
                  test.images.begin() + (size_t)(sampleIdx + 1) * imgSize, batchImgBuf.begin());
        batchLblBuf[0] = test.labels[sampleIdx];
        model.setInput(batchImgBuf.data(), 1);
        model.setLabels(batchLblBuf.data(), 1);
        model.forward(1, false);

        std::vector<float> act1(16 * 1 * 28 * 28);
        CUDA_CHECK(cudaMemcpy(act1.data(), model.conv1ActDevice(), sizeof(float) * act1.size(),
                               cudaMemcpyDeviceToHost));
        std::vector<float> act2(32 * 1 * 14 * 14);
        CUDA_CHECK(cudaMemcpy(act2.data(), model.conv2ActDevice(), sizeof(float) * act2.size(),
                               cudaMemcpyDeviceToHost));

        std::ofstream meta("results/featuremap_sample_meta.csv");
        meta << "test_index,true_label\n" << sampleIdx << "," << batchLblBuf[0] << "\n";

        std::ofstream f1("results/conv1_activations.csv");
        for (size_t i = 0; i < act1.size(); ++i) { f1 << act1[i]; if (i + 1 < act1.size()) f1 << ","; }
        f1 << "\n";

        std::ofstream f2("results/conv2_activations.csv");
        for (size_t i = 0; i < act2.size(); ++i) { f2 << act2[i]; if (i + 1 < act2.size()) f2 << ","; }
        f2 << "\n";
    }

    // -----------------------------------------------------------------------
    // Muestra cualitativa de predicciones (24 imagenes del conjunto de prueba).
    // -----------------------------------------------------------------------
    {
        std::ofstream f("results/predictions_sample.csv");
        f << "test_index,true_label,pred_label,confidence\n";
        int numQual = 24;
        const int imgSize = test.rows * test.cols;
        for (int i = 0; i < numQual; ++i) {
            int idx = testOrder[i];
            std::copy(test.images.begin() + (size_t)idx * imgSize,
                      test.images.begin() + (size_t)(idx + 1) * imgSize, batchImgBuf.begin());
            batchLblBuf[0] = test.labels[idx];
            model.setInput(batchImgBuf.data(), 1);
            model.setLabels(batchLblBuf.data(), 1);
            model.forward(1, false);
            std::vector<float> probs(10);
            CUDA_CHECK(cudaMemcpy(probs.data(), model.probsDevice(), sizeof(float) * 10,
                                   cudaMemcpyDeviceToHost));
            int pred = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
            float conf = probs[pred];
            f << idx << "," << batchLblBuf[0] << "," << pred << "," << conf << "\n";
        }
    }

    std::cout << "Entrenamiento y exportacion de resultados completados.\n";
    cudaFree(argmaxDevice);
    return 0;
}
