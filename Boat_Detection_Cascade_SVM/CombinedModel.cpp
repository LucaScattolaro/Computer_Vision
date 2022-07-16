//
// Created by luca on 16/07/21.
//

#include "CombinedModel.h"

CombinedModel::CombinedModel(string pathClassifier, string pathClassifier2, string pathSVM) {
    if ((!boatClassifier.load(pathClassifier))||(!boatClassifier2.load(pathClassifier2))) {
        cout<<endl << "ERROR: Could not load the classifiers (Check the Paths)"<<endl;
        ready = false;
    }
    else {
        svm = Algorithm::load<SVM>(pathSVM);
        hog=*(new HOGDescriptor(Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 15));

        if(svm->empty())
            ready = false;
        else
            ready = true;
    }
}

vector<BoatImage> CombinedModel::getResultTestSet(string pathsImages, bool useSVM) {

    //- Check if It's all ok with the loading of models (Cascade Classifiers and SVM)
    vector<BoatImage> boatImagesTestSet;
    try{
        if(ready) {
            //get the paths of images
            vector<string> pathsAllImages;
            glob(pathsImages + "*.*", pathsAllImages);
            for (int i = 0; i < pathsAllImages.size(); ++i) {
                //Read image and create a clone to use as clear image to show the final results of the steps
                Mat img = imread(pathsAllImages[i]);

                Mat imgClear=img.clone();
                Mat copyImg=img.clone();
                //find name of the file and name of the image
                size_t found = pathsAllImages[i].find_last_of('/');
                string nameImage = pathsAllImages[i].substr(found + 1);

                //- Create a Boat Image
                BoatImage boatImage = *(new BoatImage(nameImage, pathsImages));


                //###############    CASCADE CLASSIFIERS   ###############
                // Detect the features using Cascade of Classifiers
                vector<Rect> features,features2;
                boatClassifier.detectMultiScale(img, features, 2, 10, 0, cv::Size(24, 16));
                boatClassifier2.detectMultiScale(img, features2, 1.7, 3, 0, cv::Size(48, 24));
                //get together the ones that share a lot in common or if we have small number of boxes near each other mix together
                features.insert( features.end(), features2.begin(), features2.end() );


                //- Check How many Bounding Boxes Found the Cascade Classifiers
                bool useCannyPostProcessing=false;
                if(features.size()<20) useCannyPostProcessing=true;

                //- Translate the features in bounding boxes
                vector<BoundingBox> bboxes;
                for (int j = 0; j < features.size(); ++j) { bboxes.push_back(*(new BoundingBox(features[j]))); }


                //###############    SVM   ###############
                if (useSVM) {bboxes = chekBoxesSVM(img, bboxes);}

                //###############    POST PROCESSING   ###############
                if(useCannyPostProcessing) {
                    //Use canny to readapt the boxes and create also other boxes using canny
                    vector<BoundingBox> boxesCanny = getFinalBoxes(imgClear);
                    bboxes.insert(bboxes.end(), boxesCanny.begin(), boxesCanny.end());
                }

                //- Remove Boxes too big or too small
                vector<BoundingBox> realFinalBoxes;
                for (int j = 0; j <bboxes.size() ; ++j) {
                    if(!(bboxes[j].getArea()>(img.rows*img.cols/2))&&!(bboxes[j].getArea()<24*12)&&!(bboxes[j].getHeight()<12)&&!(bboxes[j].getWidth()<12))
                        realFinalBoxes.push_back(bboxes[j]);
                }

                //- Remove useless Boxes and Mix togheter similar ones
                realFinalBoxes = removeInnerBoxes(realFinalBoxes);
                realFinalBoxes = putTogetherSimilarBoxes(realFinalBoxes);

                //- Add the final Bounding Boxes Found
                boatImage.addBoundingBoxes(realFinalBoxes);
                boatImagesTestSet.push_back(boatImage);

            }
            //- Return the BoatImages of the Test Set obtained by the model
            return boatImagesTestSet;
        }
    }
    catch (Exception ex) {
        cout<<"Wrong Paths inserted"<<endl;
        ready=false;
    }

    // Otherwise Return an empty vector
    return vector<BoatImage>();
}
vector<BoundingBox> CombinedModel::getFinalBoxes(Mat image)
{
    //- Compute Canny
    Mat inputImgGray;
    cvtColor(image, inputImgGray, COLOR_BGR2GRAY);
    blur( inputImgGray, inputImgGray, cv::Size(9,9) );
    int lowerthresh=30;
    Canny( inputImgGray, inputImgGray, lowerthresh, lowerthresh*3, 3);

    // Get all non black points
    vector<Point> pts;
    findNonZero(inputImgGray, pts);
    // Define the radius tolerance
    int th_distance = 50;

    // Apply partition
    // All pixels within the radius tolerance distance will belong to the same class (same label)
    vector<int> labels;

    // With lambda function (require C++11)
    int th2 = th_distance * th_distance;
    int n_labels = partition(pts, labels, [th2](const Point& lhs, const Point& rhs) {
        return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < th2;
    });

    // You can save all points in the same class in a vector (one for each class), just like findContours
    vector<vector<Point>> contours(n_labels);
    for (int i = 0; i < pts.size(); ++i)
    {
        contours[labels[i]].push_back(pts[i]);
    }

    // Get bounding boxes
    vector<BoundingBox> bboxes;
    for (int i = 0; i < contours.size(); ++i)
    {
        Rect box = boundingRect(contours[i]);
        bboxes.push_back(*(new BoundingBox(box)));

    }

    return bboxes;

}

vector<BoundingBox> CombinedModel::chekBoxesSVM(Mat image, vector<BoundingBox> bboxes) {

    //- Compute canny edge detector of the image
    Mat inputImgGray;
    cvtColor(image, inputImgGray, COLOR_BGR2GRAY);
    blur( inputImgGray, inputImgGray, cv::Size(9,9) );
    int lowerthresh=20;
    Canny( inputImgGray, inputImgGray, lowerthresh, lowerthresh*3, 3);

    //- For each Bounding Box apply the SVM to check if it'a a boat or not
    vector<BoundingBox> goodBboxes;
    for (int j = 0; j < bboxes.size(); ++j) {

        //- get the part of image in the bounding Box
        Mat imageBbox;
        inputImgGray(bboxes[j].getRect()).copyTo(imageBbox);

        //- resize it in order to apply the HOG descriptor as in the training procedure of SVM
        resize(imageBbox, imageBbox, cv::Size(48, 48));

        //Compute HOG descriptor
        vector<float> descriptors;
        hog.compute(imageBbox, descriptors);
        Mat testDescriptor = cv::Mat::zeros(1, descriptors.size(), CV_32FC1);
        for (size_t i = 0; i < descriptors.size(); i++) {testDescriptor.at<float>(0, i) = descriptors[i];}

        //- Get the classification Output of SVM
        float label = svm->predict(testDescriptor);

        //if it's a boat then save it as Good Bounding Box
        if (label > 0) {goodBboxes.push_back(bboxes[j]);}

    }
    return goodBboxes;

}

vector<BoundingBox> CombinedModel::removeInnerBoxes(vector<BoundingBox> bboxes) {
    //remove boxes that are completely inside another one
    vector<BoundingBox> goodBboxes;
    for (int j = 0; j < bboxes.size(); ++j) {
        bool checkNotDone = true;
        goodBboxes.push_back(bboxes[j]);
        for (int k = 0; k < bboxes.size() && checkNotDone; ++k) {
            if ((j != k) && (bboxes[j].isContainedIn(bboxes[k]))) {
                goodBboxes.pop_back();
                checkNotDone = false;
            }
        }
    }
    return goodBboxes;
}


vector<BoundingBox> CombinedModel::putTogetherSimilarBoxes(vector<BoundingBox> boxes) {

    vector<BoundingBox> goodBboxes_UnitedIoU;
    for (int i = 0; i < boxes.size(); ++i) {goodBboxes_UnitedIoU.push_back(boxes[i]);}

    //- Check if there are boxes with a big intersection
    //  If yes create a bigger Box that contains all of them
    for (int j = 0; j < boxes.size(); ++j) {
        for (int k = 0; k < boxes.size(); ++k) {
            if((j!=k)&&(((float)boxes[j].getBoxOfIntersection(boxes[k]).getArea()/boxes[j].getArea())>0.6))
            {
                goodBboxes_UnitedIoU.push_back(boxes[j].getBoxAroundUnion(boxes[k]));
            }
        }
    }

    //- remove inner Boxes
    vector<BoundingBox> goodFinalBboxes=removeInnerBoxes(goodBboxes_UnitedIoU);

    return goodFinalBboxes;
}