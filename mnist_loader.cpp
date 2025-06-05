#include "mnist_loader.h"
#include <fstream>
#include <iostream>
#include <cstdint>
//从二进制文件中读取32位无符号整数。
static uint32_t read_uint32(std::ifstream& f) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        result <<= 8;
        result |= f.get();
    }
    return result;
}
//检查并读取图像数量、行数、列数等元数据
bool load_mnist_images(const std::string& path, std::vector<Eigen::VectorXd>& images) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening image file: " << path << std::endl;
        return false;
    }
    //读取魔数（magic number）并验证是否为2051，这是MNIST图像文件的标识
    uint32_t magic = read_uint32(file);
    if (magic != 2051) {
        std::cerr << "Invalid MNIST image file magic number." << std::endl;
        return false;
    }

    uint32_t num_images = read_uint32(file);
    uint32_t num_rows = read_uint32(file);
    uint32_t num_cols = read_uint32(file);
    int size = num_rows * num_cols;

    images.resize(num_images);

    for (uint32_t i = 0; i < num_images; ++i) {
        Eigen::VectorXd img(size);
        for (int j = 0; j < size; ++j) {
            uint8_t pixel = file.get();
            img(j) = static_cast<double>(pixel) / 255.0;
        }
        images[i] = img;
    }

    return true;
}

bool load_mnist_labels(const std::string& path, std::vector<Eigen::VectorXd>& labels, int num_classes) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening label file: " << path << std::endl;
        return false;
    }

    uint32_t magic = read_uint32(file);
    if (magic != 2049) {
        std::cerr << "Invalid MNIST label file magic number." << std::endl;
        return false;
    }

    uint32_t num_labels = read_uint32(file);
    labels.resize(num_labels);

    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t label = file.get();
        labels[i] = Eigen::VectorXd::Zero(num_classes);
        labels[i](label) = 1.0;
    }

    return true;
}
