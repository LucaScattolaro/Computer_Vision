//
// Created by luca on 30/04/21.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramicImage.h"
#include "panoramic_utils.h"

//-- Constructor
PanoramicImage::PanoramicImage(std::string pathDataset_) {
    pathDataset=pathDataset_;
    extractImagesFromDataset();
    resultImage= cv::Mat();
}

void PanoramicImage::extractImagesFromDataset() {
    // Extracting path of individual image stored in a given directory
    std::vector<std::string> images_path;
    // Path of the folder containing checkerboard images
    cv::glob(pathDataset, images_path);
    if(images_path.size()==0) throw cv::Exception();

    for (int i = 0; i < images_path.size(); ++i) {
        cv::Mat comodo = cv::imread(images_path[i]);
        imagesDataset.push_back(comodo.clone());
    }
}

cv::Mat PanoramicImage::getResultImage() {
    //-- check if the stiching has already been done
    if(cuttedImage.size()>1) {
        //-- Concatenate all the images to get the result
        cv::Mat result = cuttedImage[0];
        for (int i = 0; i < cuttedImage.size() - 1; ++i) {
            cv::hconcat(result, cuttedImage[i + 1], result);
        }
        resultImage = result.clone();
    }
    return resultImage;
}
cv::Mat PanoramicImage::getResultImageEqualized() {
    if(equalizedImages.size()>1) {
        cv::Mat result = equalizedImages[0];
        for (int i = 0; i < equalizedImages.size() - 1; ++i) {
            cv::hconcat(result, equalizedImages[i + 1], result);
        }
        resultImage = result.clone();
    }
    return resultImage;
}

void PanoramicImage::computeCylindricalProj(double angle) {
    PanoramicUtils* pu=new PanoramicUtils();
    for (int i = 0; i < imagesDataset.size(); ++i) {
        cv::Mat cilindricProjImage=pu->cylindricalProj(imagesDataset[i],angle);
        cilindricProjections.push_back(cilindricProjImage.clone());
    }

}
void PanoramicImage::detectAndCompute(cv::Ptr<cv::Feature2D> dectector) {
    for (int i = 0; i < cilindricProjections.size(); ++i) {
        std::vector<cv::KeyPoint> keypoints_comodo;
        cv::Mat descriptors_comodo=cv::Mat(0,0,CV_32F);

        //Detect and compute for each image
        dectector->detectAndCompute(cilindricProjections[i], cv::noArray(), keypoints_comodo, descriptors_comodo );
        keypoints.push_back(keypoints_comodo);
        descriptors.push_back(descriptors_comodo);
    }
}
void PanoramicImage::findGoodMatches(cv::Ptr<cv::BFMatcher> matcher, double ratio) {
    for (int i = 0; i < cilindricProjections.size()-1; ++i) {
        //-- for each image pair of images find all the matches
        std::vector<cv::DMatch> matches_comodo;
        matcher->match(descriptors[i],descriptors[i+1],matches_comodo,cv::noArray());
        matches.push_back(matches_comodo);

        //-- calculate the minimum distance among matches found between 2 images
        double min=1000;
        for (int i = 0; i < matches_comodo.size(); ++i)
            if(matches_comodo[i].distance<min)
                min=matches_comodo[i].distance;

        //-- Filter matches using the Lowest ratio test
        std::vector<cv::DMatch> good_matches_comodo;
        for (size_t i = 0; i < matches_comodo.size(); i++)
            if (matches_comodo[i].distance < ratio * min)
                good_matches_comodo.push_back(matches_comodo[i]);

        //-- Save only the good matches
        good_matches.push_back(good_matches_comodo);
    }
}
void PanoramicImage::stichingImages(double ratio) {

   if (cilindricProjections.size()==0) computeCylindricalProj(33);

    /// Detector
    cv::Ptr<cv::Feature2D> dectector;
    //here you can choose if you want to use SIFT or ORB (I decide to use sift because it gives less problems)
    dectector = cv::SIFT::create();

    /// Detect and compute
    //-- we get the descrptors and the keypoints for each image we process
    detectAndCompute(dectector);

    /// Matcher
    //-- find the matches between pair of images
    cv::Ptr<cv::BFMatcher> matcher=new cv::BFMatcher(cv::NORM_L2);
    findGoodMatches(matcher,ratio);


    /// create the images for the output
    for (int i = 0;i<cilindricProjections.size()-1; ++i) {
        //-- clone the input images in order to get a more clear code
        cv::Mat image1=cilindricProjections[i].clone();
        cv::Mat image2=cilindricProjections[i+1].clone();
        //if we are in the first iteration the image 1 should be saved in the vector used to create the output image by concatenation
        if(i==0) cuttedImage.push_back(image1);

        //show matches only for the first iteration
        if(i==0) {
            cv::Mat img_matches = cv::Mat::zeros(image1.size(), CV_8UC3);
            drawMatches(image1, keypoints[i], image2, keypoints[i + 1], good_matches[i], img_matches);
            imshow("Example of matches found", img_matches);
            cv::waitKey(0);
        }

        //-- Localize the objects in common
        std::vector<cv::Point2f> objScene_1;
        std::vector<cv::Point2f> objScene_2;
        for( size_t j = 0; j < good_matches[i].size(); j++ ){
            //-- Get the keypoints from the good matches
            objScene_1.push_back( keypoints[i][ good_matches[i][j].queryIdx].pt );
            objScene_2.push_back( keypoints[i+1][ good_matches[i][j].trainIdx].pt );
        }

        //-- find homography matrix and the average translation between 2 images
        //Try catch in order to catch exeption if the user have choosen a ratio too small
        cv::Mat inliers;
        try {
            cv::Mat H = findHomography(objScene_1, objScene_2, cv::RANSAC, 3, inliers);
        }
        catch(cv::Exception){
            std::cout<<"Ratio choosen it's too small...Please re-run the code using a bigger value"<<std::endl;
            return;
        }

        //-- Calculate the average Translation that I need to do in order to get the final result
        int avgTranslation=findAverageTranslation(inliers, i, image1.cols);

        //-- take the part of the image2 that should be concatenate to the first image
        cv::Rect boundary(avgTranslation,0,image2.cols-avgTranslation,image2.rows);
        cv::Mat secondImageCutted;
        image2(boundary).copyTo(secondImageCutted);

        cuttedImage.push_back(secondImageCutted);
    }

    //equalize images so the user can get both the normal result and also the equalized result
    equalizeImages();
}
int PanoramicImage::findAverageTranslation(cv::Mat inliers,int i,int img1_cols) {
    int avgTranslation=0,count=0;
    cv::Point2f p1,p2;
    for (int j = 0; j < good_matches[i].size(); ++j)
        if(inliers.at<bool>(j)){
            //Find the 2 points
            p1=keypoints[i][ good_matches[i][j].queryIdx].pt;
            p2=keypoints[i+1][ good_matches[i][j].trainIdx].pt;
            //calculate the distance between them
            avgTranslation+=(img1_cols-p1.x) +p2.x;
            //count the number of iteration to calculate the avg
            count++;
        }
    avgTranslation=avgTranslation/count;
    return avgTranslation;
}
void PanoramicImage::equalizeImages() {
    for (int i = 0; i < cuttedImage.size(); ++i) {
        cv::Mat image=cuttedImage[i].clone();
        //equalize the histogram
        cv::Mat hist_equalized_image;
        equalizeHist(image, hist_equalized_image);
        equalizedImages.push_back(hist_equalized_image);
    }

}













