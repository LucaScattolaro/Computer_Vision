#include <iostream>
#include "panoramicImage.h"
#include "panoramic_utils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;


int main(int argc, char** argv) {

    //-- As default we will use the kitchen dataset (if the path is ok)
    string path = "../datasets/data";
    if(argc>1) {
        cout << "The path given as argument is:: "<<argv[1] << endl;
        path=argv[1];
    }

    // Automatically Find the extension of the images
    std::vector<std::string> images_path;
    string extension;
    // Path of the folder containing checkerboard images
    cv::glob(path, images_path);
    for (int i = 1; i < images_path.size(); ++i) {
        std::size_t found = images_path[i].find_last_of(".");
        extension= images_path[i].substr(found+1);
        if(extension.size()<5)
            break;
    }
    path.append("/*.");
    path.append(extension);
    cout<<"You are using the datasaet in: "<<path<<endl;

    // try-catch used in case of wrong path!
    try {
        PanoramicImage panoramicImage = *(PanoramicImage *) (new PanoramicImage(path));
        //-- Compute Cylindrical Projections
        cout<<"Insert the value of the angle\n   (Dolomites: 27)\n   (others: 33)"<<endl;
        double angle;
        cin>>angle;
        panoramicImage.computeCylindricalProj(angle);

        //-- Stiching Images
        cout<<"Insert the ratio value to choose the good matches"<<endl;
        double ratio;
        cin>>ratio;
        panoramicImage.stichingImages(ratio);

        //-- Get the result Image
        Mat result=panoramicImage.getResultImage();
        //Mat result_equalized=panoramicImage.getResultImageEqualized();  //this gives bad results it's better to compute the equalization after

        //-- equalize the histogram of the result not equalized
        cv::Mat hist_equalized_image;
        equalizeHist(result, hist_equalized_image);

        //-- If the ratio used is too small the result image will be empty (So let's check it)
        if(!result.empty()) {
            imshow("Stiched Image", result);
            //imshow("Stiched Image Equalized", result_equalized);        //this gives bad results it's better to compute the equalization after
            imshow("Stiched Image Equalized after get the Entire Result", hist_equalized_image);
            waitKey(0);

            //-- Save the results to use them in the report
            //imwrite("panorama.png",result);
            //imwrite("panoramaEqualized.png",hist_equalized_image);

        }
    }
    catch(cv::Exception)
    {
        std::cout<<"The Path used is worng!"<<std::endl;
    }


    return 0;
}
