#include "mnist_loader.h"
#include "neural_net.h"
#include <iostream>
#include <string>

void train_model() {
    std::vector<Eigen::VectorXd> train_images, train_labels;
    std::string train_image_path = "../data/train-images-idx3-ubyte";
    std::string train_label_path = "../data/train-labels-idx1-ubyte";

    std::cout << "正在打开训练集数据: " << train_image_path << std::endl;
    std::cout << "正在打开标签集数据: " << train_label_path << std::endl;

    bool ok1 = load_mnist_images(train_image_path, train_images);
    bool ok2 = load_mnist_labels(train_label_path, train_labels, 10);

    if (!ok1 || !ok2) {
        std::cerr << "无法打开！" << std::endl;
        return;
    }

    std::cout << "加载成功 " << train_images.size() << " 个图像和 "
              << train_labels.size() << " 个标签" << std::endl;

    NeuralNetwork net(784, 128, 10);
    net.train(train_images, train_labels, 10, 0.1);
    net.save_parameters("../output/model_params.bin");
    std::cout << "模型参数已储存" << std::endl;
}

void test_model() {
    std::vector<Eigen::VectorXd> test_images, test_labels;
    std::string test_image_path = "../data/t10k-images-idx3-ubyte";
    std::string test_label_path = "../data/t10k-labels-idx1-ubyte";

    std::cout << "正在打开测试图像: " << test_image_path << std::endl;
    std::cout << "正在打开测试标签: " << test_label_path << std::endl;

    bool ok1 = load_mnist_images(test_image_path, test_images);
    bool ok2 = load_mnist_labels(test_label_path, test_labels, 10);

    if (!ok1 || !ok2) {
        std::cerr << "正在加载测试集数据" << std::endl;
        return;
    }

    std::cout << "加载成功 " << test_images.size() << " 个测试数据和 "
              << test_labels.size() << " 个测试标签" << std::endl;

    NeuralNetwork net(784, 128, 10);
    net.load_parameters("../output/model_params.bin");

    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        int pred = net.predict(test_images[i]);
        int label_index;
        test_labels[i].maxCoeff(&label_index);
        if (pred == label_index) correct++;
    }
    double accuracy = 100.0 * correct / test_images.size();
    std::cout << "测试准确率: " << accuracy << "% (" << correct << "/" << test_images.size() << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "train") {
        train_model();
    } else if (argc > 1 && std::string(argv[1]) == "try") {
        extern void run_server();
        run_server();
    } else {
        test_model();
    }
    return 0;
}
