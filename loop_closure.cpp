#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {

    // Read the images and database
    std::cout << "read database" << std::endl;
    DBoW3::Vocabulary vocab("../results/vocabulary.yml.gz");
    if (vocab.empty()) {
        std::cerr << "Vocabulary does not exist." << std::endl;
        return 1;
    }
    std::cout << "Reading images ... " << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i) {
        std::string path = "../data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // Detect ORB features
    std::cout << "detecting ORB features ..." << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat& image:images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // compare the images directly or compare one image to a database
    // images:
    std::cout << "Comparing images with images " << std::endl;
    for (int i = 0; i < images.size(); ++i) {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        // std::cout << v1 << std::endl;
        // std::cout << v1.size() << std::endl;
        for (int j = i; j < images.size(); ++j) {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            // std::cout << v2 << std::endl;
            // std::cout << v2.size() << std::endl;
            double score = vocab.score(v1, v2);
            std::cout << "Image " << i << " vs image " << j << ": " << score << std::endl;
        }
        std::cout << std::endl;
    }

    /**
     * Comparing images with images
     * <37, 0.00155759>, <39, 0.00155759>, <48, 0.00222841>, <69, 0.00222841>, <83, 0.00222841>, <101, 0.00222841>,
     * <118, 0.00155759>, <133, 0.00222841>, <164, 0.00222841>, <165, 0.00222841>, <217, 0.00222841>, <218, 0.00222841>,
     * <227, 0.00222841>, <261, 0.00222841>, <262, 0.00222841>, <266, 0.00222841>, <280, 0.00222841>, <281, 0.00222841>,
     * <302, 0.00222841>, <376, 0.00222841>, <403, 0.00222841>, <407, 0.00155759>, <438, 0.00222841>, <441, 0.00222841>,
     * ...
     * 453
     *
     * <37, 0.00155759>, <39, 0.00155759>, <48, 0.00222841>, <69, 0.00222841>, <83, 0.00222841>, <101, 0.00222841>,
     * ...
     * 453
     *
     * Image 0 vs image 0: 1
     *
     *
     * <17, 0.00230372>, <42, 0.00460744>, <57, 0.00230372>, <78, 0.00230372>, <99, 0.00161023>, <104, 0.00230372>,
     * <108, 0.00120457>, <109, 0.00161023>, <110, 0.00161023>, <115, 0.00460744>, <116, 0.00230372>, <118, 0.00161023>,
     * <119, 0.00230372>, <122, 0.00230372>, <123, 0.00460744>, <124, 0.00230372>, <141, 0.00230372>, <142, 0.00230372>,
     * <157, 0.00230372>, <159, 0.00230372>, <167, 0.00460744>, <172, 0.00230372>, <176, 0.00230372>, <203, 0.00230372>,
     * ...
     * 432
     *
     * Image 0 vs image 1: 0.0305829
     *
     *
     * <8, 0.00219351>, <25, 0.00219351>, <26, 0.00219351>, <35, 0.00219351>, <36, 0.00219351>, <58, 0.00219351>,
     * <59, 0.00219351>, <71, 0.00438701>, <72, 0.00219351>, <86, 0.00219351>, <88, 0.00219351>, <91, 0.00219351>,
     * <95, 0.00153319>, <96, 0.00219351>, <97, 0.00219351>, <100, 0.00219351>, <106, 0.00219351>, <108, 0.00114694>,
     * <134, 0.00219351>, <139, 0.00219351>, <143, 0.00219351>, <144, 0.00438701>, <152, 0.00153319>, <174, 0.00219351>,
     * ...
     * 455
     *
     * Image 0 vs image 2: 0.0221928
     * ...
     */

    // database
    std::cout << "Comparing images with database" << std::endl;
    DBoW3::Database db(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); ++i)
        db.add(descriptors[i]);
    std::cout << "database info: " << db << std::endl;
    for (int i = 0; i < descriptors.size(); ++i) {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4); // max result = 4
        std::cout << "searching for image " << i << " returns " << ret << std::endl << std::endl;
    }

    /**
     * Comparing images with database
     * database info: Database: Entries = 10, Using direct index = no. Vocabulary: k = 10, L = 5, Weighting = tf-idf, Scoring = L1-norm, Number of words = 4972
     * searching for image 0 returns 4 results:
     * <EntryId: 0, Score: 1>
     * <EntryId: 9, Score: 0.0542239>
     * <EntryId: 3, Score: 0.0308756>
     * <EntryId: 1, Score: 0.0305829>
     *
     * searching for image 1 returns 4 results:
     * <EntryId: 1, Score: 1>
     * <EntryId: 7, Score: 0.0423889>
     * <EntryId: 2, Score: 0.0399516>
     * <EntryId: 3, Score: 0.0340341>
     * ...
     *
     */


    std::cout << "done" << std::endl;

    return 0;
}