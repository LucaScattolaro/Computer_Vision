//
// Created by luca on 04/04/21.
//

#ifndef LAB3_FILTER_H
#define LAB3_FILTER_H

#endif //LAB3_FILTER_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter{

// Methods

public:
    // constructor
    // input_img: image to be filtered
    // filter_size : size of the kernel/window of the filter
    Filter();
    Filter(cv::Mat input_img, int filter_size);

    //virtual // perform filtering (in base class do nothing, to be reimplemented in the derived filters)
    virtual void doFilter();

    // get the output of the filter
    cv::Mat getResult();

    //set the window size (square window of dimensions size x size)
    void setSize(int size);

    //get the Window Size
    int getSize();

// Data
protected:
    // input image
    cv::Mat input_image;
    // output image (filter result)
    cv::Mat result_image;
    // window size
    int filter_size;

};

// Gaussian Filter
class GaussianFilter : public Filter  {
    public:
        // place constructor
        GaussianFilter(cv::Mat input_img, int size,double sigma_);
        // re-implement  doFilter()
        void doFilter() override;
        void setSigma(double sigma_);
    private:
        // additional parameter: standard deviation (sigma)
        double sigma;
};

class MedianFilter : public Filter {
    public:
        // place constructor
        MedianFilter(cv::Mat input_img, int size);
        // re-implement  doFilter()
        void doFilter() override;

};

class BilateralFilter : public Filter {
    public:
        // place constructor
        BilateralFilter(cv::Mat input_img, int size,double sigma_range,double sigma_space);
        // re-implement  doFilter()
        void doFilter() override;
        void setSigmaRange(double sigma_range);
        void setsigmaSpace(double sigma_space);
    private:
        // additional parameters: sigma_range, sigma_space
        double sigmaRange;
        double sigmaSpace;
};
