#include "mnist.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>

static uint32_t read_be_u32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

MnistDataset load_mnist(const std::string& imagesPath, const std::string& labelsPath) {
    MnistDataset ds;

    std::ifstream imgFile(imagesPath, std::ios::binary);
    if (!imgFile) throw std::runtime_error("No se pudo abrir " + imagesPath);
    uint32_t magicImg = read_be_u32(imgFile);
    if (magicImg != 2051) throw std::runtime_error("Magic number invalido en " + imagesPath);
    uint32_t numImages = read_be_u32(imgFile);
    uint32_t rows = read_be_u32(imgFile);
    uint32_t cols = read_be_u32(imgFile);

    std::ifstream lblFile(labelsPath, std::ios::binary);
    if (!lblFile) throw std::runtime_error("No se pudo abrir " + labelsPath);
    uint32_t magicLbl = read_be_u32(lblFile);
    if (magicLbl != 2049) throw std::runtime_error("Magic number invalido en " + labelsPath);
    uint32_t numLabels = read_be_u32(lblFile);
    if (numLabels != numImages) throw std::runtime_error("Desajuste numImages/numLabels");

    ds.numImages = (int)numImages;
    ds.rows = (int)rows;
    ds.cols = (int)cols;

    size_t pixCount = (size_t)numImages * rows * cols;
    std::vector<unsigned char> rawPix(pixCount);
    imgFile.read(reinterpret_cast<char*>(rawPix.data()), pixCount);

    std::vector<unsigned char> rawLbl(numLabels);
    lblFile.read(reinterpret_cast<char*>(rawLbl.data()), numLabels);

    ds.images.resize(pixCount);
    for (size_t i = 0; i < pixCount; ++i) ds.images[i] = rawPix[i] / 255.0f;

    ds.labels.resize(numLabels);
    for (size_t i = 0; i < numLabels; ++i) ds.labels[i] = (int)rawLbl[i];

    return ds;
}
