#include "util.h"
#include <cmath>
#include <algorithm>

Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
    return 1.0 / (1.0 + (-z.array()).exp());
}

Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& z) {
    Eigen::VectorXd s = sigmoid(z);
    return s.array() * (1 - s.array());
}

Eigen::VectorXd softmax(const Eigen::VectorXd& z) {
    Eigen::ArrayXd exp_z = (z.array() - z.maxCoeff()).exp(); // 防止溢出
    return exp_z / exp_z.sum();
}

double cross_entropy_loss(const Eigen::VectorXd& predicted,
                          const Eigen::VectorXd& actual) {
    const double epsilon = 1e-12;
    return - (actual.array() * (predicted.array() + epsilon).log()).sum();
}

Eigen::VectorXd one_hot(int label, int num_classes) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(num_classes);
    v(label) = 1.0;
    return v;
}

int argmax(const Eigen::VectorXd& vec) {
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return static_cast<int>(maxIndex);
}
