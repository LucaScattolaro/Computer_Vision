//
// Created by luca on 17/07/21.
//

#ifndef PROJECTCV_DATASETPREPARATION_UTILS_H
#define PROJECTCV_DATASETPREPARATION_UTILS_H

#include <iostream>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "BoundingBox.h"
#include "BoatImage.h"
#include "CombinedModel.h"

using namespace std;
using namespace cv;
using namespace ml;



class DatasetPreparation {
public:

    //- Functions that I used to create datasets for Models and to perform training of SVM
    void createDatasetForSVM(string trainingSetImagesPath, string trainingSetInfoPath, string boatClassifierPath,string cartellaPositiveForSVM, string cartellaNegativeForSVM)
    {
        try {//- Create and Load the Cascade Classifier
            CascadeClassifier boatClassifier;
            if (!boatClassifier.load(boatClassifierPath)){ cout << "ERROR: Could not load the classifier (check the paths)";}
            else{

                //- Read the file boats.info (easier to process in this case)
                vector<BoatImage> boatimages;

                ifstream labelsFile(trainingSetInfoPath);
                if (labelsFile.is_open()) {
                    //-Process all the Lines (1 Line = 1 Image with Boats)
                    string str;
                    while (getline(labelsFile, str)) {
                        //- Get the name of the Image
                        vector<string> values=split(str,' ');
                        string nameImage=split(values[0],'/')[1];

                        //- Create the Boat Image that will contain the image and the bounding boxes of the Ground Truth
                        BoatImage boat=*(new BoatImage(nameImage,trainingSetImagesPath));
                        int offset=2;
                        for (int i = 0; i < stoi(values[1]); ++i) {
                            boat.addBoundingBox(*(new BoundingBox(stoi(values[offset+i]),stoi(values[offset+i+1]),stoi(values[offset+i+2]),stoi(values[offset+i+3]))));
                            offset=offset+3;
                        }
                        boatimages.push_back(boat);
                    }
                }

                cout<<" Process: Creation of dataset for SVM -> Start"<<endl;
                for (int i = 0; i < boatimages.size(); ++i) {
                    Mat image=boatimages[i].getImage();
                    Mat imagecopy=image.clone();
                    string nameImage=boatimages[i].getNameNoExtension();
                    vector<BoundingBox> bboxesImage=boatimages[i].getBoundingBoxes();

                    //Apply Smoothing and Canny Edge Detector
                    Mat inputImgGray;
                    cvtColor(image, inputImgGray, COLOR_BGR2GRAY);
                    blur( inputImgGray, inputImgGray, cv::Size(9,9) );
                    int lowerthresh=20;
                    Canny( inputImgGray, inputImgGray, lowerthresh, lowerthresh*3, 3);


                    //###############    POSITIVE   ###############
                    //-Create all the positive images for the training set of the SVM
                    //- for each image check each Ground Truth's Bounding Box and save it in the directory for the SVM
                    for (int j = 0; j < bboxesImage.size(); ++j) {
                        string nameImageBox=nameImage+"_bb"+to_string(j);

                        //- Get the Bounding Boxes of the Ground Truth
                        Mat imageBoundingBox;
                        inputImgGray(bboxesImage[j].getRect()).copyTo(imageBoundingBox);
                        string nameNewImageBoundingBox=cartellaPositiveForSVM+nameImageBox+"_0.png";
                        imwrite(nameNewImageBoundingBox, imageBoundingBox);

                        //- Take a subImage of the Bounding Box that contains a Boat (in order to increase the number of positive images of Training Set)
                        Mat imageCutted_1;
                        int x=(int)(((float)imageBoundingBox.cols)*((float)1/5));
                        int y=(int)(((float)imageBoundingBox.rows)*((float)1/5));
                        int width=(int)(((float)imageBoundingBox.cols)*((float)3/5));
                        int height=(int)(((float)imageBoundingBox.rows)*((float)3/5));
                        imageBoundingBox(Rect(x,y,width,height)).copyTo(imageCutted_1);
                        string nameNewImage_1=cartellaPositiveForSVM+nameImageBox+"_1.png";
                        imwrite(nameNewImage_1, imageCutted_1);


                        //- Take another subImage of the Bounding Box that contains a Boat (in order to increase the number of positive images of Training Set)
                        Mat imageCutted_2;
                        x=(int)(((float)imageBoundingBox.cols)*((float)2/5));
                        y=(int)(((float)imageBoundingBox.rows)*((float)2/5));
                        width=(int)(((float)imageBoundingBox.cols)*((float)1/5));
                        height=(int)(((float)imageBoundingBox.rows)*((float)1/5));
                        imageBoundingBox(Rect(x,y,width,height)).copyTo(imageCutted_2);
                        string nameNewImage_2=cartellaPositiveForSVM+nameImageBox+"_2.png";
                        imwrite(nameNewImage_2, imageCutted_2);

                    }


                    //###############    NEGATIVE   ###############
                    //- Detect the Bounding Boxes using the Cascade Classifier
                    vector<Rect> features;
                    boatClassifier.detectMultiScale(image, features, 2, 4,0, cv::Size(48, 48));

                    //- find False Positive Bounding Boxes
                    vector<Rect> badFeatures =boatimages[i].getUselessBboxes(features);
                    int cont=0;
                    for (auto&& feature : badFeatures) {
                        cont++;
                        Mat imageCutted;
                        inputImgGray(feature).copyTo(imageCutted);
                        string nameNewImage=cartellaNegativeForSVM+boatimages[i].getNameNoExtension()+"_"+to_string(cont)+".png";
                        imwrite(nameNewImage, imageCutted);
                    }


                }
                cout<<" Process: Creation of dataset for SVM -> End"<<endl<<endl;
            }
        }
        catch (Exception ex) {
            cout<<"Wrong path inserted"<<endl;
        }


    }

    void train_svm_hog(string negativeBboxes,string poitiveBboxes,string pathSavingSVM)
    {
        cout<<" Process: Load Images -> Start"<<endl;

        //HOG detector, used to calculate the HOG descriptor
        HOGDescriptor hog(Size(48, 48), Size(16, 16), Size(8, 8), Size(8, 8), 15);
        //The dimension of the HOG descriptor is determined by the size of the picture, the size of the detection window, the block size, and the number of bins in the histogram of the cell unit
        int DescriptorDim;

        //Set SVM parameters
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::Types::C_SVC);
        svm->setC(0.1);
        svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10000, 1e-6));

        //- Get the paths of all images (Positive and Negative)
        vector<string> pathsAllPositiveImages;
        vector<string> pathsAllNegativeImages;
        glob(poitiveBboxes, pathsAllPositiveImages);
        glob(negativeBboxes, pathsAllNegativeImages);

        //- Print the Number of images Found
        cout<<"     -Number of images Positive: "<<pathsAllPositiveImages.size()<<endl;
        cout<<"     -Number of images Negative: "<<pathsAllNegativeImages.size()<<endl;
        int numImages=pathsAllPositiveImages.size()+pathsAllNegativeImages.size();

        //A matrix of feature vectors of all training samples, the number of rows is equal to the number of all samples, the number of columns is equal to the dimension of the HOG descriptor
        Mat sampleFeatureMat;
        //Category vector of training samples, the number of rows is equal to the number of all samples, the number of columns is equal to 1; 1 means there is a target, -1 means no target
        Mat sampleLabelMat;

        //Read positive sample images in turn to generate HOG descriptors
        for (int num = 0; num < pathsAllPositiveImages.size(); num++)
        {
            Mat image = imread(pathsAllPositiveImages[num]);
            resize(image, image, Size(48, 48));
            //HOG descriptor vector
            vector<float> descriptors;
            //Calculate the HOG descriptor and detect the moving step of the window (8,8)
            hog.compute(image, descriptors, Size(8, 8));

            //Initialize the eigenvector matrix and category matrix when processing the first sample, because the eigenvector matrix can only be initialized if the dimension of the eigenvector is known
            if (0 == num)
            {
                //Dimension of HOG descriptor
                DescriptorDim = descriptors.size();
                //Initialize the matrix of feature vectors of all training samples, the number of rows is equal to the number of all samples, the number of columns is equal to the dimension of HOG descriptor sub sampleFeatureMat
                sampleFeatureMat = Mat::zeros(numImages, DescriptorDim, CV_32FC1);
                //Initialize the category vector of training samples, the number of rows is equal to the number of all samples, the number of columns is equal to 1
                sampleLabelMat = Mat::zeros(numImages, 1, CV_32SC1);
            }
            //Copy the calculated HOG descriptor to the sample feature matrix sampleFeatureMat
            for (int i = 0; i < DescriptorDim; i++)
            {
                //The ith element in the feature vector of the num sample
                sampleFeatureMat.at<float>(num, i) = descriptors[i];
            }
            //The positive sample category is 1 = Boat
            sampleLabelMat.at<float>(num, 0) = 1;

        }

        //Read negative sample pictures in turn to generate HOG descriptors
        for (int num = 0; num < pathsAllNegativeImages.size(); num++)
        {
            Mat src = imread(pathsAllNegativeImages[num]);
            resize(src, src, cv::Size(48, 48));

            //HOG descriptor vector
            vector<float> descriptors;
            //Calculate the HOG descriptor and detect the moving step of the window (8,8)
            hog.compute(src, descriptors, Size(8, 8));


            //Copy the calculated HOG descriptor to the sample feature matrix sampleFeatureMat
            for (int i = 0; i < DescriptorDim; i++)
            {
                //The ith element in the feature vector of the PosSamNO+num samples
                sampleFeatureMat.at<float>(num + pathsAllPositiveImages.size(), i) = descriptors[i];
            }
            //Negative sample category is -1 = No Boat
            sampleLabelMat.at<float>(num + pathsAllPositiveImages.size(), 0) = -1;
        }
        cout<<" Process: Load Images -> End"<<endl<<endl;

        if(numImages>0) {
            //Train SVM classifier
            cout << " Process: training SVM classifier -> Start" << std::endl;
            Ptr<TrainData> td = TrainData::create(sampleFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);
            svm->train(td);
            cout << " Process: training SVM classifier -> End" << std::endl;

            //Save the trained SVM model as an xml file
            svm->save(pathSavingSVM);
            cout << " Process: Training SVM -> End" << endl;
        }
        else
        {
            cout<<"No Images Found!"<<endl;
        }

    }

    void createDatasetForCascadeClassifier(string trainingSetPath,string labeltxtPath,string fileposName,string filenegName,string path_pos,string path_neg)
    {
        //- Get the paths of all training set images
        vector<string> pathsAllImages;
        try {
            glob(trainingSetPath, pathsAllImages);
            cout<<"Number of images in Training Set Found: "<<pathsAllImages.size()<<endl;

            //- Create the files as output stream in order to write all the information
            ofstream outfile_pos (fileposName);

            for (int i = 0; i < pathsAllImages.size(); ++i) {
                //find name of the file and name of the image
                size_t found = pathsAllImages[i].find_last_of('/');
                string nameImage=pathsAllImages[i].substr(found+1);
                size_t found_end = nameImage.find_last_of('.');
                string nameFile=nameImage.substr(0,found_end);
                nameFile=nameFile+".txt";

                // Read the image
                Mat img=imread(pathsAllImages[i]);

                // Look for the file related to the image
                ifstream labelsFile(labeltxtPath+nameFile);
                if (labelsFile.is_open()) {
                    //process all data in order to save the file boat.info
                    //Open an existing file
                    string str;
                    vector<string> lines;
                    while (getline(labelsFile, str)) {lines.push_back(str);}

                    //- Write into boats.info in the right way described below:
                    //  pos/[NameImage] [Number of Bounding Boxes] [info for each BoundingBox.....]
                    string pathPosToWriteonFile="pos/";
                    vector<vector<string>> values;
                    string stringToWrite=pathPosToWriteonFile+nameImage+" "+to_string(lines.size());
                    for (int i = 0; i < lines.size(); ++i) {
                        vector<string> comodo;
                        comodo=split(lines[i],':');
                        vector<string> bboxes=split(comodo[1],';');;
                        int xmin=stoi(bboxes[0]);
                        int xmax=stoi(bboxes[1]);
                        int ymin=stoi(bboxes[2]);
                        int ymax=stoi(bboxes[3]);
                        stringToWrite=stringToWrite+" "+to_string(xmin)+" "+to_string(ymin)+" "+to_string(xmax-xmin)+" "+to_string(ymax-ymin);

                    }
                    outfile_pos<<stringToWrite<<endl;

                    //- Save the image into the positive directory (We have to do that to train the cascade of classifiers)
                    string pathImgPos=path_pos+nameImage;
                    bool check = imwrite(pathImgPos,img.clone());
                    if (check == false) {cout << "Saving of positive image: FAILED (check the paths given)" << endl;}
                }

            }

            // close the file
            outfile_pos.close();

            cout<<"Processing Done For Positive Images"<<endl;
            cout<<"Now fill the negative directory with other negative images if you want\nand then press a key to continue the creation of bg.txt"<<endl;
            string comodo;
            cin>>comodo;


            //- Get the paths of all training set Negative images
            vector<string> pathsAllNegativeImages;
            glob(path_neg+"*.*", pathsAllNegativeImages);
            cout<<"Number of images in Negative Set: "<<pathsAllNegativeImages.size()<<endl;

            //- Create the files as output stream in order to write all the information
            ofstream outfile_neg (filenegName);

            for (int i = 0; i < pathsAllNegativeImages.size(); ++i) {
                //find name of the file and name of the image
                size_t found = pathsAllNegativeImages[i].find_last_of('/');
                string nameImage = pathsAllNegativeImages[i].substr(found + 1);

                // Read the image
                string stringToWriteNeg="neg/"+nameImage;
                outfile_neg<<stringToWriteNeg<<endl;
            }

            // close the file
            outfile_neg.close();
        }
        catch (Exception ex) {
            cout<<"Wrong Paths"<<endl;
        }

    }


    // Function Useful to perform splitting of string given a char
    vector<string> split(const string &s, char delimiter) {
        vector<string> tokens;
        string token;
        istringstream tokenStream(s);
        while (getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

};


#endif //PROJECTCV_DATASETPREPARATION_UTILS_H
