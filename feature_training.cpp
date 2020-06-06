#include <iostream>
#include <vector>
#include <string>
#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

int main(int argc, char** argv) {

    // Read images
    std::cout << "Reading images ..." << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i) {
        std::string path = "../data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // Detect ORB features
    std::cout << "Detecting ORB features ..." << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat& image:images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // create vocabulary
    std::cout << "Creating vocabulary ..." << std::endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    std::cout << "Vocabulaty info: " << vocab << std::endl;
    vocab.save("../results/vocabulary.yml.gz");
    std::cout << "done" << std::endl;

    return 0;
}
