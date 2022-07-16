//
// Created by luca on 30/04/21.
//

#ifndef LAB5_PANORAMICIMAGE_H
#define LAB5_PANORAMICIMAGE_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

class PanoramicImage{
// Methods
public:
    // constructor
    PanoramicImage(std::string pathDataset_);
    //-- Methods
    void computeCylindricalProj(double angle);
    void stichingImages(double ratio);
    cv::Mat getResultImage();
    cv::Mat getResultImageEqualized();

// Private
private:
    // Data
    std::string pathDataset;
    std::vector<cv::Mat> imagesDataset;
    cv::Mat resultImage;

    // I used also these variable in case in the future I need them for other computations
    std::vector<cv::Mat> cilindricProjections;
    std::vector<cv::Mat> cuttedImage;
    std::vector<cv::Mat> equalizedImages;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::vector<cv::DMatch>> good_matches;

    //-- Methods
    void equalizeImages();
    void extractImagesFromDataset();
    void detectAndCompute(cv::Ptr<cv::Feature2D> dectector);
    void findGoodMatches(cv::Ptr<cv::BFMatcher> matcher,double ratio);
    int findAverageTranslation(cv::Mat inliers,int i,int img1_cols);

};
#endif //LAB5_PANORAMICIMAGE_H
