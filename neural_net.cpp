#include "neural_net.h"
#include "util.h"
#include <random>
#include <iostream> 
#include <iomanip> // for output formatting
#include <fstream>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size)
{
    // 初始化权重与偏置（使用正态分布）
    std::random_device rd; // 随机数生成器
    std::mt19937 gen(rd()); // 使用随机数种子
    std::normal_distribution<> dist(0, 1.0); // 正态分布，均值0，标准差1

    W1 = Eigen::MatrixXd(hidden_size, input_size); // hidden_size 行 input_size 列
    b1 = Eigen::VectorXd::Zero(hidden_size); //列向量，大小为 hidden_size
    W2 = Eigen::MatrixXd(output_size, hidden_size);
    b2 = Eigen::VectorXd::Zero(output_size);
    // Xavier 初始化

    for (int i = 0; i < hidden_size; ++i)
        for (int j = 0; j < input_size; ++j)
            W1(i, j) = dist(gen) * std::sqrt(1.0 / input_size);  // Xavier-like init

    for (int i = 0; i < output_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            W2(i, j) = dist(gen) * std::sqrt(1.0 / hidden_size);
}
//向前传播
Eigen::VectorXd NeuralNetwork::forward(const Eigen::VectorXd& input) {
    // input: [784]
    Eigen::VectorXd Z1 = W1 * input + b1;      // [hidden_size]
    Eigen::VectorXd A1 = sigmoid(Z1);          // 激活
    Eigen::VectorXd Z2 = W2 * A1 + b2;         // [output_size]
    Eigen::VectorXd A2 = softmax(Z2);          // softmax 输出概率
    return A2;
}
//获得预测值
int NeuralNetwork::predict(const Eigen::VectorXd& input) {
    Eigen::VectorXd output = forward(input);
    return argmax(output);
}
//X_train: 输入数据，其中每一列代表一个样本的784个像素点；y_train: 标签数据 
// epochs: 训练轮数，learning_rate: 学习率
void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& X_train,
                          const std::vector<Eigen::VectorXd>& y_train,
                          int epochs, double learning_rate) {
    int n_samples = X_train.size();
//重复训练
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        int correct = 0; // 记录正确预测的数量
        // 遍历每个样本
        // X_train[i]: 表示第i个样本，是一个列向量；y_train[i]: 标签数据
        for (int i = 0; i < n_samples; ++i) {
            const Eigen::VectorXd& x = X_train[i];
            const Eigen::VectorXd& y = y_train[i]; // one-hot标签

            // 向前传播
            Eigen::VectorXd Z1 = W1 * x + b1;
            Eigen::VectorXd A1 = sigmoid(Z1);
            Eigen::VectorXd Z2 = W2 * A1 + b2;
            Eigen::VectorXd A2 = softmax(Z2);

            // 损失函数
            total_loss += cross_entropy_loss(A2, y);
            //如果预测正确，即 A2 的最大值索引与 y 的最大值索引相同，则正确计数加1
            if (argmax(A2) == argmax(y)) correct++;

            // 反向传播
            Eigen::VectorXd dZ2 = A2 - y; // 输出层误差
            Eigen::MatrixXd dW2 = dZ2 * A1.transpose(); //对W2的梯度
            Eigen::VectorXd db2 = dZ2; //对b2的梯度

            Eigen::VectorXd dA1 = W2.transpose() * dZ2;
            Eigen::VectorXd dZ1 = dA1.array() * sigmoid_derivative(Z1).array(); // elementwise
            Eigen::MatrixXd dW1 = dZ1 * x.transpose();
            Eigen::VectorXd db1 = dZ1;

            // === Gradient Descent Step ===
            W2 -= learning_rate * dW2;
            b2 -= learning_rate * db2;
            W1 -= learning_rate * dW1;
            b1 -= learning_rate * db1;
        }

        // 每个 epoch 打印损失和准确率
        std::cout << "Epoch " << epoch + 1
                  << " | 损失: " << std::fixed << std::setprecision(4) << total_loss / n_samples
                  << " | 准确率: " << (100.0 * correct / n_samples) << "%" << std::endl;
    }
}

void NeuralNetwork::save_parameters(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "无法打开文件保存参数: " << filename << std::endl;
        return;
    }
    int rows, cols;
    // 保存 W1
    rows = W1.rows();
    cols = W1.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(W1.data()), sizeof(double) * rows * cols);
    // 保存 b1
    rows = b1.size();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(b1.data()), sizeof(double) * rows);
    // 保存 W2
    rows = W2.rows();
    cols = W2.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(W2.data()), sizeof(double) * rows * cols);
    // 保存 b2
    rows = b2.size();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(b2.data()), sizeof(double) * rows);
    out.close();
}

void NeuralNetwork::load_parameters(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "无法打开文件加载参数: " << filename << std::endl;
        return;
    }
    int rows, cols;
    // 读取 W1
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    W1.resize(rows, cols);
    in.read(reinterpret_cast<char*>(W1.data()), sizeof(double) * rows * cols);
    // 读取 b1
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    b1.resize(rows);
    in.read(reinterpret_cast<char*>(b1.data()), sizeof(double) * rows);
    // 读取 W2
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    W2.resize(rows, cols);
    in.read(reinterpret_cast<char*>(W2.data()), sizeof(double) * rows * cols);
    // 读取 b2
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    b2.resize(rows);
    in.read(reinterpret_cast<char*>(b2.data()), sizeof(double) * rows);
    in.close();
}
