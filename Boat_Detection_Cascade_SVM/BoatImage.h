//
// Created by luca on 10/07/21.
//

#ifndef PROJECTCV_BOATIMAGE_H
#define PROJECTCV_BOATIMAGE_H



#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "BoundingBox.h"

using namespace std;
using namespace cv;

class BoatImage {
private:
    string nameImage;
    vector<BoundingBox> bboxes;
    string pathImage;

public:
    BoatImage(string name,string path);
    void addBoundingBox(BoundingBox b);
    void addBoundingBox(Rect b);
    void addBoundingBoxes(vector<BoundingBox> boxes);
    void addBoundingBoxes(vector<Rect> boxes);
    int getNumBoundingBoxes();
    string getPath();
    string getInfo();
    string getName();
    string getNameNoExtension();
    vector<Rect> getUselessBboxes(vector<Rect> features);
    Mat getImage();
    Mat getImageWithBoxes();
    vector<BoundingBox> getBoundingBoxes();
    string getIoU(BoatImage grundTruth);

    void setNameImage( string nameImage);

    void setPathImage( string pathImage);


};


#endif //PROJECTCV_BOATIMAGE_H
