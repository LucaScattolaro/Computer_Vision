//
// Created by luca on 10/07/21.
//

#ifndef PROJECTCV_BOUNDINGBOX_H
#define PROJECTCV_BOUNDINGBOX_H


#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

class BoundingBox {
private:
    int x;
    int y;
    int width;
    int height;
    vector<Point> verteces;
    float goodness;

public:
    BoundingBox(int x,int y,int width,int height);
    BoundingBox(int x,int y,int width,int height,float goodness);
    BoundingBox(Rect rect);
    BoundingBox(Rect rect,float goodness);

    // Getter
    int getX();
    int getY();
    int getHeight();
    int getWidth();
    String getInfo();
    Rect getRect();
    vector<Point> getVerteces();
    float getArea();
    float getGoodness();
    BoundingBox getBoxOfIntersection(BoundingBox b);
    BoundingBox getBoxAroundUnion(BoundingBox b);

    // Setter
    void setGoodness(float value);
    // Others
    float IntersectionOverUnion(BoundingBox b);
    bool isContainedIn(BoundingBox b);
    bool hasIntersectionWith(BoundingBox b);

};




#endif //PROJECTCV_BOUNDINGBOX_H
