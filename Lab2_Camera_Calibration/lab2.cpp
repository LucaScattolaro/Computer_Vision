#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // Path of the folder containing checkerboard images
    string path = "../data/checkerboard_images/*.png";
    string path_testImg ="../data/test_image.png";
    if(argc>2) {
        cout << "The path of checkerboard Images given as argument is: "<<argv[1] << endl;
        cout << "The path of Test Image given as argument is: "<<argv[2] << endl;
        String comodo_Check=argv[1];
        path=comodo_Check.append("/*.png");
        String comodo_testImage=argv[2];
        path_testImg=comodo_testImage;
    }

    //int checkboardSize[2]={6,5};  //width height
    Size checkboardSize=Size(6,5);
    double squareSize=0.11;

    // Creating vector to store vectors of 3D points for each checkerboard image
    vector<vector<Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    vector<vector<Point2f> > imgpoints;


    // Defining the world coordinates for 3D points
    vector<Point3f> objp;
    for(int i=0; i<checkboardSize.height; i++)
        for(int j=0; j<checkboardSize.width; j++)
            objp.push_back(Point3f(j*squareSize,i*squareSize,0));

    ///#####      1. Loads the checkerboard images
    // Extracting path of individual image stored in a given directory
    vector<String> images_path;
    glob(path, images_path);
    Mat img, grayImg,outImg;



    ///#####      2. Detects the checkerboard intersections per image.
    // vector to store the pixel coordinates of detected checker board corners
    vector<Point2f> corner_pts;
    bool foundCorners;

    for(int i=0; i<images_path.size(); i++)
    {
        img = imread(images_path[i]);
        if( img.empty() ){
            cout << "Could not open or find the checkerboard images! (Check the path)\n" << endl;
            return -1;
        }
        cvtColor(img,grayImg,cv::COLOR_BGR2GRAY);
        // Finding checker board corners
        // If desired number of corners are found in the image then success = true
        foundCorners = findChessboardCorners(grayImg,checkboardSize, corner_pts);
        if(foundCorners)
        {
            TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            // refining pixel coordinates for given 2d points.
            cornerSubPix(grayImg,corner_pts,Size(5,5), Size(-1,-1),criteria);
            // Displaying the detected corner points on the checker board
            drawChessboardCorners(img,checkboardSize, corner_pts, foundCorners);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        if(i<3) {
            Mat comodo;
            resize(img, comodo, Size(img.cols / 2.0, img.rows / 2.0));
            imshow("Corner Visualization", comodo);
            waitKey(0);

        }
    }
    Size sizeImages=Size(grayImg.cols,grayImg.rows);


    ///#####      3. Calibrates the camera by using the intersections found.
    Mat cameraMatrix,distCoeffs,rvec,tvec;
    calibrateCamera(objpoints, imgpoints, Size(grayImg.cols,grayImg.rows), cameraMatrix, distCoeffs, rvec, tvec);

    ///#####      4. Print to output the estimated intrinsic and distortion parameters with correct names
    cout << "Rotation vector :\n" << rvec << endl<<endl;
    cout << "Translation vector :\n" << tvec << endl<<endl;
    cout << "Intrinsic Parameters :\n" << cameraMatrix << endl<<endl;
    cout << "Distorsion Coefficients :\n " << distCoeffs << endl<<endl;

    ///#####      5. Computes the mean reprojection error
    double reprojectionErrors[objpoints.size()],min=1000,max=-1000,mean_error;
    vector<Point2f> projectedPoints;
    int i_bestImage,i_worst_path;

    for (int i = 0; i < objpoints.size(); ++i) {
        projectPoints(objpoints[i], rvec.row(i), tvec.row(i), cameraMatrix, distCoeffs, projectedPoints);
        //calculate reprojection Errors
        // RIGHT WAY
        double dist=0;
        for (int j = 0; j < objpoints[i].size(); ++j)
            dist = dist + norm(imgpoints[i][j] - projectedPoints[j]);
        mean_error+=reprojectionErrors[i]=dist/projectedPoints.size();

        // WRONG WAY
        //mean_error+=reprojectionErrors[i]=norm(imgpoints[i], projectedPoints, NORM_L2)/(projectedPoints.size());

        //take min and max value of the error so then we can choose the best and worst image
        if (reprojectionErrors[i] < min) {
            min = reprojectionErrors[i];    i_bestImage=i;
        }
        if (reprojectionErrors[i] > max) {
            max = reprojectionErrors[i];    i_worst_path=i;
        }
        //cout<<"image "<<i<<": "<<reprojectionErrors[i]<<endl;
    }

    ///#####      6. Choosing among the input images, prints the names of the image for which the calibration performs best and the image for which it performs worst. What is the parameter you are using to perform this choice?
    mean_error=mean_error/objpoints.size();
    cout<<"MEAN TOTAL ERROR "<<mean_error<<endl;
    cout<<"BEST IMAGE "<<images_path[i_bestImage]<<"   "<<min<<endl;
    cout<<"WORST IMAGE "<<images_path[i_worst_path]<<"   "<<max<<endl;

    ///#####      7. Undistorts and rectifies a new image acquired with the same camera
    // Build the undistort map
    Mat map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs,cv::Mat(), cameraMatrix, Size(grayImg.cols,grayImg.rows),CV_16SC2, map1, map2);
    Mat imageUndistorted, inputImage = imread(path_testImg);
    if( inputImage.empty() ){
        cout << "Could not open or find the test image! (Check the path)\n" << endl;
        return -1;
    }
    cv::remap(inputImage, imageUndistorted, map1, map2, cv::INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar());

    ///#####      8. Compare the result in a split view using the highgui module.
    Mat outputImage;
    hconcat(inputImage,imageUndistorted,outputImage);
    resize(outputImage,outputImage,Size(outputImage.cols/2.0,outputImage.rows/2.0));
    imshow("Result", outputImage);
    waitKey(0);
    imwrite("result.jpg", outputImage);

    return 0;
}