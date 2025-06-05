#include "web_server.h"
#include "neural_net.h"
#include "crow_all.h"
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// base64 解码
static std::string base64_decode(const std::string &in);

// 将 base64 PNG 数据转为 28x28 Eigen::VectorXd
static Eigen::VectorXd png_base64_to_vector(const std::string& base64_png);

void run_server() {
    NeuralNetwork net(784, 128, 10);
    
    // 尝试多个可能的模型路径
    std::string model_path;
    if (std::ifstream("../output/model_params.bin").good()) {
        model_path = "../output/model_params.bin";
    } else if (std::ifstream("output/model_params.bin").good()) {
        model_path = "output/model_params.bin";
    } else if (std::ifstream("./model_params.bin").good()) {
        model_path = "./model_params.bin";
    } else {
        std::cerr << "找不到模型参数文件！请确保已训练模型。" << std::endl;
        return;
    }
    
    std::cout << "加载模型参数: " << model_path << std::endl;
    net.load_parameters(model_path);
    crow::SimpleApp app;

    CROW_ROUTE(app, "/")([](){
        return R"html(<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>手写数字识别</title>
<style>
body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
#canvas { border: 2px solid #333; background: #fff; cursor: crosshair; }
button { margin: 10px; padding: 10px 20px; font-size: 16px; }
#result { font-size: 24px; font-weight: bold; color: #007bff; margin: 20px; }
</style>
</head>
<body>
<h1>🔢 手写数字识别</h1>
<p>在下面的画布上画一个数字（0-9），然后点击识别按钮</p>
<canvas id="canvas" width="280" height="280"></canvas><br>
<button onclick="clearCanvas()">🗑️ 清空</button>
<button onclick="predict()">🔍 识别</button>
<div id="result">请在画布上画一个数字</div>
<script>
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let drawing = false;

// 初始化画布背景为白色
ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, 280, 280);

canvas.onmousedown = e => { 
    drawing = true; 
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    // 绘制一个小点，防止单击时没有绘制
    ctx.fillStyle = '#000';
    ctx.fillRect(e.offsetX-1, e.offsetY-1, 2, 2);
};
canvas.onmouseup = e => { drawing = false; };
canvas.onmousemove = e => {
    if (!drawing) return;
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
};

function clearCanvas() {
    ctx.clearRect(0, 0, 280, 280);
    // 重新填充白色背景
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, 280, 280);
    document.getElementById('result').innerText = '请在画布上画一个数字';
}

function predict() {
    let img = canvas.toDataURL('image/png');
    document.getElementById('result').innerText = '识别中...';
    
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: img })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => Promise.reject(err));
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('result').innerText = '识别结果: ' + data.result;
    })
    .catch(error => {
        if (error.error) {
            document.getElementById('result').innerText = error.error;
        } else {
            document.getElementById('result').innerText = '识别失败，请重试';
        }
        console.error('Error:', error);
    });
}
</script>
</body>
</html>)html";
    });

    CROW_ROUTE(app, "/predict").methods("POST"_method)
    ([&net](const crow::request& req){
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400);
        std::string img_base64 = body["image"].s();
        // 去掉 data:image/png;base64, 前缀
        size_t pos = img_base64.find(",");
        if (pos != std::string::npos) img_base64 = img_base64.substr(pos+1);
        
        Eigen::VectorXd input = png_base64_to_vector(img_base64);
        crow::json::wvalue res;
        
        // 检查特殊返回值
        if (input(0) == -1) {
            res["error"] = "图像处理失败";
            return crow::response(400, res);
        } else if (input(0) == -2) {
            res["error"] = "请画一个清晰的数字再识别";
            return crow::response(400, res);
        }
        
        int pred = net.predict(input);
        std::cout << "神经网络预测结果: " << pred << std::endl;
        res["result"] = pred;
        return crow::response{res};
    });

    std::cout << "请在浏览器打开 http://127.0.0.1:18080/ 进行手写数字识别体验" << std::endl;
    app.port(18080).multithreaded().run();
}

// 简化的 base64 解码实现
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
static std::string base64_decode(const std::string &input) {
    BIO *bio, *b64;
    int decodeLen = input.length() * 3 / 4;
    std::string result(decodeLen, '\0');
    
    bio = BIO_new_mem_buf(input.data(), -1);
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);
    
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    int len = BIO_read(bio, &result[0], input.length());
    BIO_free_all(bio);
    
    if (len > 0) {
        result.resize(len);
    } else {
        result.clear();
    }
    return result;
}

// base64 PNG -> 28x28 Eigen::VectorXd
static Eigen::VectorXd png_base64_to_vector(const std::string& base64_png) {
    try {
        std::string png_data = base64_decode(base64_png);
        if (png_data.empty()) {
            std::cerr << "Base64 解码失败" << std::endl;
            return Eigen::VectorXd::Constant(784, -1); // 用-1表示错误
        }
        
        std::vector<uchar> buf(png_data.begin(), png_data.end());
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
        
        if (img.empty()) {
            std::cerr << "PNG 解码失败" << std::endl;
            return Eigen::VectorXd::Constant(784, -1);
        }
        
        std::cout << "原始图像尺寸: " << img.rows << "x" << img.cols << std::endl;
        
        // 检查是否为空白图像（几乎全白或全黑）
        cv::Scalar mean_val_orig = cv::mean(img);
        std::cout << "原始图像均值: " << mean_val_orig[0] << std::endl;
        
        // 如果图像几乎全白（背景）或全黑，认为是空白图像
        if (mean_val_orig[0] > 250 || mean_val_orig[0] < 5) {
            std::cout << "检测到空白图像，拒绝识别" << std::endl;
            return Eigen::VectorXd::Constant(784, -2); // 用-2表示空白图像
        }
        
        // 寻找内容边界框，裁剪掉多余的空白部分
        cv::Mat binary;
        cv::threshold(img, binary, 128, 255, cv::THRESH_BINARY_INV);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            std::cout << "未找到有效内容，拒绝识别" << std::endl;
            return Eigen::VectorXd::Constant(784, -2);
        }
        
        // 找到最大的轮廓（假设是数字）
        cv::Rect boundingBox = cv::boundingRect(contours[0]);
        for (size_t i = 1; i < contours.size(); ++i) {
            cv::Rect rect = cv::boundingRect(contours[i]);
            if (rect.area() > boundingBox.area()) {
                boundingBox = rect;
            }
        }
        
        std::cout << "内容边界框: " << boundingBox.x << "," << boundingBox.y 
                  << " " << boundingBox.width << "x" << boundingBox.height << std::endl;
        
        // 添加一些边距
        int margin = 10;
        boundingBox.x = std::max(0, boundingBox.x - margin);
        boundingBox.y = std::max(0, boundingBox.y - margin);
        boundingBox.width = std::min(img.cols - boundingBox.x, boundingBox.width + 2*margin);
        boundingBox.height = std::min(img.rows - boundingBox.y, boundingBox.height + 2*margin);
        
        // 裁剪到内容区域
        cv::Mat cropped = img(boundingBox);
        
        // 创建正方形图像，保持宽高比
        int maxDim = std::max(cropped.rows, cropped.cols);
        cv::Mat square = cv::Mat::ones(maxDim, maxDim, CV_8UC1) * 255; // 白色背景
        
        int offsetX = (maxDim - cropped.cols) / 2;
        int offsetY = (maxDim - cropped.rows) / 2;
        cropped.copyTo(square(cv::Rect(offsetX, offsetY, cropped.cols, cropped.rows)));
        
        // 调整到 28x28
        cv::resize(square, img, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
        
        // 反转颜色：Canvas是黑字白底，MNIST是白字黑底
        cv::bitwise_not(img, img);
        
        // 转换为 double 类型，归一化到 [0,1]
        img.convertTo(img, CV_64F, 1.0/255.0);
        
        // 计算最终图像统计信息
        cv::Scalar mean_val = cv::mean(img);
        double min_val, max_val;
        cv::minMaxLoc(img, &min_val, &max_val);
        std::cout << "最终图像统计 - 均值: " << mean_val[0] << ", 最小值: " << min_val << ", 最大值: " << max_val << std::endl;
        
        // 转为 Eigen::VectorXd
        Eigen::VectorXd v(784);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                v(i * 28 + j) = img.at<double>(i, j);
            }
        }
        
        // 检查是否有有效的像素变化
        double variance = 0;
        double mean = mean_val[0];
        for (int i = 0; i < 784; ++i) {
            variance += (v(i) - mean) * (v(i) - mean);
        }
        variance /= 784;
        std::cout << "像素方差: " << variance << std::endl;
        
        if (variance < 0.01) {
            std::cout << "图像变化太小，可能是无效输入" << std::endl;
            return Eigen::VectorXd::Constant(784, -2);
        }
        
        return v;
    } catch (const std::exception& e) {
        std::cerr << "图像处理异常: " << e.what() << std::endl;
        return Eigen::VectorXd::Constant(784, -1);
    }
} 