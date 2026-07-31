#pragma once
#include <string>
#include <vector>

struct MnistDataset {
    std::vector<float> images;  // [numImages*rows*cols], normalizado a [0,1]
    std::vector<int> labels;    // [numImages]
    int numImages = 0;
    int rows = 0;
    int cols = 0;
};

MnistDataset load_mnist(const std::string& imagesPath, const std::string& labelsPath);
