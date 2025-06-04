#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>
#include <Eigen/Dense>

bool load_mnist_images(const std::string& path, std::vector<Eigen::VectorXd>& images);
bool load_mnist_labels(const std::string& path, std::vector<Eigen::VectorXd>& labels, int num_classes = 10);

#endif
