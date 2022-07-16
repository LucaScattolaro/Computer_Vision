//
// Created by luca on 10/07/21.
//

#include "BoatImage.h"

//- CONSTRUCTORS
BoatImage::BoatImage(string name,string path) {
    pathImage=path;
    nameImage=name;
}

//- GETTER
int BoatImage::getNumBoundingBoxes() {
    return bboxes.size();
}

string BoatImage::getInfo() {
    // Return a string with all the info of the BoatImage instance
    string ret="Image: "+nameImage+"  BoundingBoxes: "+to_string(bboxes.size())+"    ->   ";
    for (int i = 0; i < bboxes.size(); ++i) {
        ret=ret+bboxes[i].getInfo()+"  ";
    }
    return ret;
}


string BoatImage::getName() {
    return nameImage;
}

Mat BoatImage::getImage() {
    // Return the input image
    Mat image=imread(pathImage+nameImage);
    if (image.empty()) //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded check the Path..!!" << endl;
        return Mat();
    }
    return image;
}

Mat BoatImage::getImageWithBoxes() {
    // Return the input image with the Bounding Boxes drawn on it
    Mat image=imread(pathImage+nameImage);
    if (image.empty()) //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded check the Path..!!" << endl;
        return Mat();
    }

    //- Return the image with the Bounding Boxes drawn on it with different colors with respect to the value of the goodness=IoU
    for (auto&& boxex : bboxes) {
        Scalar color;
        if(boxex.getGoodness()==0)
            color=cv::Scalar(0, 0, 255);
        else if(boxex.getGoodness()==1)
            color=cv::Scalar(0, 255, 0);
        else{
            if(boxex.getGoodness()>0.6)
                color=cv::Scalar(0, 255, 0);
            else if((boxex.getGoodness()<=0.6)&&(boxex.getGoodness()>=0.3))
                color=cv::Scalar(0, 255, 255);
            else
                color=cv::Scalar(0, 165, 255);

            cv::putText(image, to_string(boxex.getGoodness()), Point(boxex.getX(), boxex.getY()), FONT_HERSHEY_DUPLEX, 1,color, 2, false);
        }
        cv::rectangle(image, boxex.getRect(), color, 2);
    }
    return image;
}

vector<Rect> BoatImage::getUselessBboxes(vector<Rect> features) {
    // Find all the Wrong Bounding Boxes -> The ones that don't have intersections with the ground Truth
    vector<Rect> uselessBboxes;
    for (int i = 0; i < features.size(); ++i) {
        BoundingBox b=*(new BoundingBox(features[i]));
        bool notIntersection=true;
        for (int j = 0; j < bboxes.size()&&notIntersection; ++j) {
            if(bboxes[j].hasIntersectionWith(b))
                notIntersection=false;
        }
        if(notIntersection)
            uselessBboxes.push_back(b.getRect());
    }


    return uselessBboxes;
}

string BoatImage::getNameNoExtension() {
    // Return the name of the image without the extension of the file
    vector<string> tokens;
    string token;
    istringstream tokenStream(nameImage);
    while (getline(tokenStream, token, '.')) {
        tokens.push_back(token);
    }
    return tokens[0];
}


string BoatImage::getIoU(BoatImage groundTruth) {
    //Return a string of the information about the difference of the BoatImage get by my model with the Boat Image of the Ground Truth
    vector<BoundingBox> grundTruthBoxes=groundTruth.getBoundingBoxes();
    float iouMyBoxes[grundTruthBoxes.size()][2];
    float IoU[bboxes.size()][grundTruthBoxes.size()];


    // For each Bounding Box calculate the Intersection Over Union w.r.t all the Bounding boxes of the Ground Truth
    for (int i = 0; i < bboxes.size(); ++i) {
        for (int j = 0; j < grundTruthBoxes.size(); ++j) {
            IoU[i][j]=bboxes[i].IntersectionOverUnion(grundTruthBoxes[j]);
        }
    }

    // For each Bounding box of the ground truth find the optimal Bounding Box found by the model
    for (int i = 0; i < grundTruthBoxes.size(); ++i) {
        float max=0.0;
        int indexMax=-1;
        for (int j = 0; j < bboxes.size(); ++j) {
            if(IoU[j][i]>max)
            {
                max=IoU[j][i];
                indexMax=j;
            }

        }
        iouMyBoxes[i][0]=indexMax;
        iouMyBoxes[i][1]=max;
    }

    // Remove duplicates (one box found by the model could not cover more than one box of the Ground truth)
    int cont=0;
    for (int i = 0; i < grundTruthBoxes.size(); ++i) {
        bool finito=false;
        if(iouMyBoxes[i][0]!=-1)
            for (int j = i+1; j < grundTruthBoxes.size()&&(!finito); ++j) {
                if(iouMyBoxes[i][0]==iouMyBoxes[j][0])
                {
                    if(iouMyBoxes[i][1]>iouMyBoxes[j][1]) {
                        iouMyBoxes[j][1]=0;
                        iouMyBoxes[j][0]=-1;
                    }
                    else
                    {
                        iouMyBoxes[i][1]=0;
                        iouMyBoxes[i][0]=-1;
                        finito=true;
                        cont=cont+1;
                    }
                }
            }
        else
            cont=cont+1;
    }

    //- Create the string that contain all the information of the comparison
    string result="Image: "+this->getName()
                  +"\nBounding Boxes Found By Model:  "+to_string(bboxes.size())
                  +"\nBounding Boxes Ground Truth  :  "+to_string(grundTruthBoxes.size())
                  +"\n Boats Found:\n";
    for (int i = 0; i < grundTruthBoxes.size(); ++i) {
        if(iouMyBoxes[i][0]!=-1) {
            result = result + " - IoU: " + to_string(iouMyBoxes[i][1]) + "   RealBox: " + grundTruthBoxes[i].getInfo() +" BoxFound: " + bboxes[iouMyBoxes[i][0]].getInfo() + "\n";
            bboxes[iouMyBoxes[i][0]].setGoodness(iouMyBoxes[i][1]);
        }
    }
    result=result+" Missing Boats: "+to_string(cont)+"\n";
    result=result+" False Positive: "+to_string(bboxes.size()-(grundTruthBoxes.size()-cont));
    return result;
}

vector<BoundingBox> BoatImage::getBoundingBoxes() {
    return bboxes;
}

string BoatImage::getPath() {
    return pathImage;
}


//- SETTER
void BoatImage::setNameImage(string nameImage) {
    this->nameImage = nameImage;
}

void BoatImage::setPathImage(string pathImage) {
    this->pathImage = pathImage;
}



//_ ADD BOUNDING BOXES
void BoatImage::addBoundingBox(BoundingBox b) {
    bboxes.push_back(b);
}

void BoatImage::addBoundingBox(Rect b) {
    bboxes.push_back(*(new BoundingBox(b)));

}

void BoatImage::addBoundingBoxes(vector<BoundingBox> boxes) {
    for (int i = 0; i < boxes.size(); ++i) {
        addBoundingBox(boxes[i]);
    }

}

void BoatImage::addBoundingBoxes(vector<Rect> boxes) {
    for (int i = 0; i < boxes.size(); ++i) {
        addBoundingBox(boxes[i]);
    }
}





