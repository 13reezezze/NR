#include "mnist_loader.h"
#include "neural_net.h"
#include <iostream>
#include <string>

void train_model() {
    std::vector<Eigen::VectorXd> train_images, train_labels;
    std::string train_image_path = "../data/train-images-idx3-ubyte";
    std::string train_label_path = "../data/train-labels-idx1-ubyte";

    std::cout << "Attempting to open train image file: " << train_image_path << std::endl;
    std::cout << "Attempting to open train label file: " << train_label_path << std::endl;

    bool ok1 = load_mnist_images(train_image_path, train_images);
    bool ok2 = load_mnist_labels(train_label_path, train_labels, 10);

    if (!ok1 || !ok2) {
        std::cerr << "Failed to load MNIST train data" << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << train_images.size() << " images and "
              << train_labels.size() << " labels" << std::endl;

    NeuralNetwork net(784, 128, 10);
    net.train(train_images, train_labels, 10, 0.1);
    net.save_parameters("../output/model_params.bin");
    std::cout << "Model parameters saved to ../output/model_params.bin" << std::endl;
}

void test_model() {
    std::vector<Eigen::VectorXd> test_images, test_labels;
    std::string test_image_path = "../data/t10k-images-idx3-ubyte";
    std::string test_label_path = "../data/t10k-labels-idx1-ubyte";

    std::cout << "Attempting to open test image file: " << test_image_path << std::endl;
    std::cout << "Attempting to open test label file: " << test_label_path << std::endl;

    bool ok1 = load_mnist_images(test_image_path, test_images);
    bool ok2 = load_mnist_labels(test_label_path, test_labels, 10);

    if (!ok1 || !ok2) {
        std::cerr << "Failed to load MNIST test data" << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << test_images.size() << " test images and "
              << test_labels.size() << " test labels" << std::endl;

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
    std::cout << "Test Accuracy: " << accuracy << "% (" << correct << "/" << test_images.size() << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "train") {
        train_model();
    } else {
        test_model();
    }
    return 0;
}
