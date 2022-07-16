//
// Created by luca on 04/04/21.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

// constructor
Filter::Filter(cv::Mat input_img, int size) {

    input_image = input_img;
    if (size % 2 == 0)
        size++;
    filter_size = size;
}

// for base class do nothing (in derived classes it performs the corresponding filter)
void Filter::doFilter() {

    // it just returns a copy of the input image
    result_image = input_image.clone();

}

// get output of the filter
cv::Mat Filter::getResult() {

    return result_image;
}

//set window size (it needs to be odd)
void Filter::setSize(int size) {

    if (size % 2 == 0)
        size++;
    filter_size = size;
}

//get window size
int Filter::getSize() {

    return filter_size;
}

Filter::Filter() {
    input_image = cv::Mat();
    filter_size = 0;
}

// Write your code to implement the Gaussian, median and bilateral filters
MedianFilter::MedianFilter(cv::Mat input_img, int size) : Filter(input_img, size) {
}

void MedianFilter::doFilter() {
    Filter::doFilter();
    cv::medianBlur(input_image,result_image,filter_size);
}

BilateralFilter::BilateralFilter(cv::Mat input_img, int size,double sigma_range,double sigma_space) : Filter(input_img, size) {
    sigmaRange=sigma_range;
    sigmaSpace=sigma_space;

}

void BilateralFilter::doFilter() {
    Filter::doFilter();
    cv::bilateralFilter(input_image,result_image,filter_size,sigmaRange,sigmaSpace);

}

void BilateralFilter::setSigmaRange(double sigma_range) {
    sigmaRange = sigma_range;
}

void BilateralFilter::setsigmaSpace(double sigma_space) {
    sigmaSpace = sigma_space;
}

GaussianFilter::GaussianFilter(cv::Mat input_img, int size,double sigma_) : Filter(input_img, size) {
    sigma= sigma_;
}

void GaussianFilter::doFilter() {
    Filter::doFilter();
    cv::GaussianBlur(input_image,result_image,cv::Size(filter_size,filter_size),sigma);
}

void GaussianFilter::setSigma(double sigma_) {
    sigma= sigma_;
}
