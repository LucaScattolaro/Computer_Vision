#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include "filter.h"
using namespace std;
using namespace cv;


/// MEDIAN FILTER - TRACKBAR
int kernel_size;
int kernel_size_max;
void onChangeTrackbarMedian(int, void* userFilt)
{
    MedianFilter medianFilt=*(MedianFilter*)userFilt;
    medianFilt.setSize(kernel_size);
    medianFilt.doFilter();
    Mat result=medianFilt.getResult();
    imshow("MedianFilter", result);
}

/// GAUSSIAN FILTER - TRACKBAR
int kernel_size_G;
int kernel_size_max_G;
int sigma;
int sigma_max;
void onChangeTrackbarGaussian(int, void* userFilt)
{
    GaussianFilter gaussianFilt=*(GaussianFilter*)userFilt;
    gaussianFilt.setSize(kernel_size_G);
    gaussianFilt.setSigma(sigma);
    gaussianFilt.doFilter();
    Mat result=gaussianFilt.getResult();
    imshow("GaussianFilter", result);
}

/// BILATERAL FILTER - TRACKBAR
int kernel_size_B=7;
int sigma_range;
int sigma_space;
int sigma_max_B;
void onChangeTrackbarBilateral(int, void* userFilt)
{
    BilateralFilter bilateralFilt=*(BilateralFilter*)userFilt;
    //set all the parameters
    bilateralFilt.setSize(kernel_size_B);
    bilateralFilt.setSigmaRange(sigma_range);
    bilateralFilt.setsigmaSpace(sigma_space);
    //apply the filter
    bilateralFilt.doFilter();
    //show result
    Mat result=bilateralFilt.getResult();
    imshow("BilateralFilter", result);
}

Mat showHistogram(vector<cv::Mat>& hists);

int main(int argc, char** argv) {
    /// 1. Loads an image (e.g., one of the provided images like “image.jpg” or “countryside.jpg”)
    string path = "../data/*";
    if(argc>1) {
        cout << "The path given as argument is:: "<<argv[1] << endl;
        String comodo=argv[1];
        path=comodo.append("/*");;
    }
    vector<String> images_path;

    glob(path, images_path);
    cout<<"choose 1 image among:"<<endl;
    for (int i = 0; i < images_path.size(); ++i)
        cout<<" "<<i<<". "<<images_path[i]<<endl;
    //choose the image to work with
    int indexImg;
    cin>>indexImg;
    if((indexImg<0||indexImg>3)||(indexImg>=images_path.size()))
        indexImg=images_path.size()-1;

    Mat img, grayImg,outImg;
    Mat inputImg=imread(images_path[indexImg]);
    if( inputImg.empty() ){
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    //check if the image choosen is too big, normal or small so after we can choose the better way to visualize it
    bool needResizeImg=inputImg.cols>1500;
    bool tooSmallImg=inputImg.cols<600;

    /// 2. Prints the histograms of the image. You have to compute 3 histograms, one for each channel (i.e., R, G and B) with 256 bins and [0, 255] as range.
    vector<Mat> bgr_planes,bgr_hist;
    split( inputImg, bgr_planes );
    int histSize = 256;
    float range[] = { 0, 256 };                     //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;        //default values
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    bgr_hist.push_back(b_hist);
    bgr_hist.push_back(g_hist);
    bgr_hist.push_back(r_hist);
    Mat histoImg1=showHistogram(bgr_hist);

    /// 3. Equalizes the R,G and B channels by using cv::equalizeHist()
    Mat equalized_image,equalized_image_planes_b,equalized_image_planes_g,equalized_image_planes_r;
    vector<Mat> equalized_bgr_planes,bgr_hist_eq;
    equalizeHist(bgr_planes[0], equalized_image_planes_b);
    equalizeHist(bgr_planes[1], equalized_image_planes_g);
    equalizeHist(bgr_planes[2], equalized_image_planes_r);
    equalized_bgr_planes.push_back(equalized_image_planes_b);
    equalized_bgr_planes.push_back(equalized_image_planes_g);
    equalized_bgr_planes.push_back(equalized_image_planes_r);
    merge(equalized_bgr_planes,equalized_image);

    /// 4. Shows the equalized image and the histogram of its channels.
    split( equalized_image, equalized_bgr_planes );
    Mat b_hist_eq, g_hist_eq, r_hist_eq;
    calcHist( &equalized_bgr_planes[0], 1, 0, Mat(), b_hist_eq, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &equalized_bgr_planes[1], 1, 0, Mat(), g_hist_eq, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &equalized_bgr_planes[2], 1, 0, Mat(), r_hist_eq, 1, &histSize, &histRange, uniform, accumulate );
    bgr_hist_eq.push_back(b_hist_eq);
    bgr_hist_eq.push_back(g_hist_eq);
    bgr_hist_eq.push_back(r_hist_eq);
    Mat histoImg2=showHistogram(bgr_hist_eq);

    /// 5.Try to perform some experiments to obtain a better equalization using a different color space, e.g. the HSV color space
    Mat input_hsv,output_hsv,outputRgbImg,equalized_image_plane[3],equalized_hsv_image;
    vector<Mat> hsv_eq_planes;

    // convert to HSV color space
    cvtColor(inputImg, input_hsv, COLOR_BGR2HSV);
    vector<Mat> hsv_planes,hsv_hist;
    split( input_hsv, hsv_planes);
    equalizeHist(hsv_planes[0], equalized_image_plane[0]);
    equalizeHist(hsv_planes[1], equalized_image_plane[1]);
    equalizeHist(hsv_planes[2], equalized_image_plane[2]);

    //Let the user Choose which channel to equalize
    int channelChoosen;
    cout<<"HSV equalization, chose:      (if you type a wrong number the default channel chosen will be V)\n  0. H\n  1. S\n  2. V"<<endl;
    cin>>channelChoosen;
    if(!(channelChoosen<3 && channelChoosen>-1)){
        channelChoosen=2;}
    for (int i = 0; i < 3; ++i) {
        if (i == channelChoosen)
            hsv_eq_planes.push_back(equalized_image_plane[channelChoosen]);
        else
            hsv_eq_planes.push_back(hsv_planes[i]);
    }

    // Re-construct the equalized image
    merge(hsv_eq_planes,equalized_hsv_image);
    //Change the color space from HSV to RGB
    cvtColor(equalized_hsv_image, outputRgbImg, cv::COLOR_HSV2BGR);

    //calculate the histogram of the BGR channels of the "HSV - equalized" image
    Mat b_hist_eqhsv, g_hist_eqhsv, r_hist_eqhsv;
    vector<Mat> hsv_bgr_planes;
    split( outputRgbImg, hsv_bgr_planes);
    calcHist( &hsv_bgr_planes[0], 1, 0, Mat(), b_hist_eqhsv, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &hsv_bgr_planes[1], 1, 0, Mat(), g_hist_eqhsv, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &hsv_bgr_planes[2], 1, 0, Mat(), r_hist_eqhsv, 1, &histSize, &histRange, uniform, accumulate );
    hsv_hist.push_back(b_hist_eqhsv);
    hsv_hist.push_back(g_hist_eqhsv);
    hsv_hist.push_back(r_hist_eqhsv);
    Mat histoImg3=showHistogram(hsv_hist);


    /// Visualization of results

    //Let the user choose the Visualization Modality
    int visualizationType;
    cout<<"How do you want to see the Images of Results and Histograms? Type:\n  0. I don't want to see anything\n  1. All togheter\n  2. One image for Result"<<endl;
    cin>>visualizationType;
    if(visualizationType==1)
    {
        //Images
        Mat intermediateImage,outputImage;
        hconcat(inputImg,equalized_image,intermediateImage);
        hconcat(intermediateImage,outputRgbImg,outputImage);
        //some images need to be resized in order to be viewed on screen in a reasonable way
        if(needResizeImg)
            resize(outputImage,outputImage,Size(outputImage.cols/3.0,outputImage.rows/3.0));
        else if(!tooSmallImg)
            resize(outputImage,outputImage,Size(outputImage.cols/1.5,outputImage.rows/1.5));
        //Histograms
        Mat intermediateImageHisto,outputImageHisto;
        hconcat(histoImg1,histoImg2,intermediateImageHisto);
        hconcat(intermediateImageHisto,histoImg3,outputImageHisto);
        //Show everything Together
        Mat output;
        resize(outputImageHisto,outputImageHisto,Size(outputImage.cols,outputImage.cols*outputImageHisto.rows/outputImageHisto.cols));
        vconcat(outputImageHisto,outputImage,output);
        imshow("input BGR + histogram ||| equalized BGR + histogram ||| equalized HSV-BGR channel "+to_string(channelChoosen)+" + histogram",output);
        waitKey(0);

        //save the image
        string name = images_path[indexImg].substr(images_path[indexImg].find("data/")+5,images_path[indexImg].size()); // token is "scott"
        name=name.substr(0,name.find("."));
        imwrite(name+"_result.jpg", output); // A JPG FILE IS BEING SAVED
    }
    else if(visualizationType==2) {
        Mat outImg1, outImg2, outImg3;
        Mat outputRgbImg_comodo;
        //some images need to be resized in order to be viewed on screen in a reasonable way
        if (needResizeImg){
            resize(inputImg, inputImg, Size(inputImg.cols / 2.0, inputImg.rows / 2.0));
            resize(equalized_image, equalized_image, Size(equalized_image.cols / 2.0, equalized_image.rows / 2.0));
            resize(outputRgbImg,outputRgbImg_comodo,Size(outputRgbImg.cols/2.0,outputRgbImg.rows/2.0));
        }
        else
            outputRgbImg_comodo = outputRgbImg;

        //input BGR - histogram
        resize(histoImg1,histoImg1,Size(inputImg.cols,inputImg.cols*histoImg1.rows/histoImg1.cols));
        vconcat(histoImg1,inputImg,outImg1);
        imshow("input BGR - histogram",outImg1);
        //equalized BGR - histogram
        resize(histoImg2,histoImg2,Size(equalized_image.cols,equalized_image.cols*histoImg2.rows/histoImg2.cols));
        vconcat(histoImg2,equalized_image,outImg2);
        imshow("equalized BGR - histogram",outImg2);
        //equalized HSV-BGR - histogram
        resize(histoImg3,histoImg3,Size(outputRgbImg_comodo.cols,outputRgbImg_comodo.cols*histoImg3.rows/histoImg3.cols));
        vconcat(histoImg3,outputRgbImg_comodo,outImg3);
        imshow("equalized HSV-BGR - histogram",outImg3);
        waitKey(0);
    }

    /// Part 2: Image Filtering
    //some images need to be resized in order to be viewed on screen in a reasonable way
    if (needResizeImg)
        resize(outputRgbImg,outputRgbImg,Size(outputRgbImg.cols/2.0,outputRgbImg.rows/2.0));
    Filter* filter_;
    int filtType;
    do {
        //let the user choose which Filter to use
        cout<<"Chose Type of Filter:"<<"\n  0: Exit\n  1: Median \n  2: Gaussian \n  3: Bilateral"<<endl;
        cin>>filtType;
        switch(filtType)
        {
            /// median filter
            case 1:
                filter_=new MedianFilter(outputRgbImg,0);
                kernel_size = 0;
                kernel_size_max = 255;
                namedWindow("MedianFilter", 1);
                //make trackbar call back
                createTrackbar("Kernel Size", "MedianFilter", &kernel_size, kernel_size_max, onChangeTrackbarMedian,(void*)filter_);
                imshow("MedianFilter", outputRgbImg);
                waitKey(0);break;

            /// gaussian filter
            case 2:
                filter_=new GaussianFilter(outputRgbImg,0,0);
                sigma=0;
                sigma_max=255;
                kernel_size_G = 0;
                kernel_size_max_G = 255;
                namedWindow("GaussianFilter", 1);
                //make trackbar call back
                createTrackbar("Kernel Size", "GaussianFilter", &kernel_size_G, kernel_size_max_G, onChangeTrackbarGaussian,(void*)filter_);
                createTrackbar("Sigma", "GaussianFilter", &sigma, sigma_max, onChangeTrackbarGaussian,(void*)filter_);
                imshow("GaussianFilter", outputRgbImg);
                waitKey(0);break;
            /// bilateral filter

            case 3:
                filter_=new BilateralFilter(outputRgbImg,0,0,0);
                sigma_space=0;
                sigma_range=0;
                sigma_max_B=255;
                namedWindow("BilateralFilter", 1);
                //make trackbar call back
                createTrackbar("Sigma Space", "BilateralFilter", &sigma_space, sigma_max_B, onChangeTrackbarBilateral,(void*)filter_);
                createTrackbar("Sigma Range", "BilateralFilter", &sigma_range, sigma_max_B, onChangeTrackbarBilateral,(void*)filter_);
                imshow("BilateralFilter", outputRgbImg);
                waitKey(0);break;
            default: break;
        }
    }while(filtType>0 && filtType<4);
    return 0;
}

Mat showHistogram(vector<cv::Mat>& hists)
{
    // Min/Max computation
    double hmax[3] = {0,0,0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);
    std::string wname[3] = { "blue", "green", "red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());
    // Display each histogram in a canvas
    Mat outputImage,intermediateImg;
    for (int i = 0, end = hists.size(); i < end; i++){
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);
        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++){
            cv::line(canvas[i],
                     cv::Point(j, rows),
                     cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
                    hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
                    1, 8, 0);
        }
    }
    hconcat(canvas[0],canvas[1],intermediateImg);
    hconcat(intermediateImg,canvas[2],outputImage);
    return outputImage;
}
