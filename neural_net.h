#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Eigen/Dense>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);

    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    int predict(const Eigen::VectorXd& input); // 返回预测数字
    void train(const std::vector<Eigen::VectorXd>& X_train,
           const std::vector<Eigen::VectorXd>& y_train,
           int epochs, double learning_rate);
    void save_parameters(const std::string& filename) const;
    void load_parameters(const std::string& filename);

private:
    int input_size, hidden_size, output_size;

    Eigen::MatrixXd W1; // 输入层 -> 隐藏层
    Eigen::VectorXd b1;

    Eigen::MatrixXd W2; // 隐藏层 -> 输出层
    Eigen::VectorXd b2;

};

#endif
