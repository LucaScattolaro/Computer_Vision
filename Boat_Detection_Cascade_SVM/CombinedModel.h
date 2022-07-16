//
// Created by luca on 16/07/21.
//

#ifndef PROJECTCV_COMBINEDMODEL_H
#define PROJECTCV_COMBINEDMODEL_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "BoundingBox.h"
#include "BoatImage.h"

using namespace std;
using namespace cv;
using namespace ml;

class CombinedModel {
private:
    CascadeClassifier boatClassifier;
    CascadeClassifier boatClassifier2;
    Ptr<SVM> svm;
    HOGDescriptor hog;
    bool ready;

public:
    CombinedModel(string pathClassifier,string pathClassifier2, string pathSVM);
    vector<BoatImage> getResultTestSet(string pathsImages,bool useSVM);

private:
    vector<BoundingBox> chekBoxesSVM(Mat image,vector<BoundingBox> bboxes);
    vector<BoundingBox> removeInnerBoxes(vector<BoundingBox> bboxes);
    vector<BoundingBox> putTogetherSimilarBoxes(vector<BoundingBox> goodBboxes_bboxespostRedimension);
    vector<BoundingBox> getFinalBoxes(Mat image);

};


#endif //PROJECTCV_COMBINEDMODEL_H
