#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>

#include <vector>
//激活函数
Eigen::VectorXd sigmoid(const Eigen::VectorXd& z);
//激活函数导数
Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& z);
//softmax函数，表示数字0到9的概率
Eigen::VectorXd softmax(const Eigen::VectorXd& z);

//损失函数
double cross_entropy_loss(const Eigen::VectorXd& predicted,
                          const Eigen::VectorXd& actual);

//工具函数，将标签转换为独热编码
Eigen::VectorXd one_hot(int label, int num_classes = 10);

//返回向量的最大值索引
int argmax(const Eigen::VectorXd& vec);

#endif
