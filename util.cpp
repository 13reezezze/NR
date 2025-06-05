#include "util.h"
#include <cmath>
#include <algorithm>
//激活函数 sigmoid(z) = 1 / (1+exp(-z))
Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
    return 1.0 / (1.0 + (-z.array()).exp());
}
//激活函数的导数 Dsigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& z) {
    Eigen::VectorXd s = sigmoid(z);
    return s.array() * (1 - s.array());
}
//softmax函数 softmax(z) = exp(z) / sum(exp(z))
Eigen::VectorXd softmax(const Eigen::VectorXd& z) {
    Eigen::ArrayXd exp_z = (z.array() - z.maxCoeff()).exp(); // 防止溢出
    return exp_z / exp_z.sum();
}
//交叉熵损失函数 
// L(y, y_hat) = -sum(y * log(y_hat))
double cross_entropy_loss(const Eigen::VectorXd& predicted,
                          const Eigen::VectorXd& actual) {
    const double epsilon = 1e-12;
    return - (actual.array() * (predicted.array() + epsilon).log()).sum();
}
//one-hot标签
Eigen::VectorXd one_hot(int label, int num_classes) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(num_classes);
    v(label) = 1.0;
    return v;
}
//返回最有可能的数
int argmax(const Eigen::VectorXd& vec) {
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return static_cast<int>(maxIndex);
}
