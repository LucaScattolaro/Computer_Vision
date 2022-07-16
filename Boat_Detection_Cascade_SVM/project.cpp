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
#include "datasetPreparation_utils.h"


using namespace std;
using namespace cv;
using namespace ml;

//- Functions
void testTheFinalModel(string testSetPath,string testSetLabelsPath);

//- Global variable to use utility functions
DatasetPreparation* datsetPreparation=new DatasetPreparation();


int main(int argc, char** argv) {

    cout<<"MENU:\n";
    cout<<" 1- Create the dataset for the training of the cascade of Classifiers (Not necessary because the project already contains the .xml of cascade)"<<endl;
    cout<<" 2- Create Dataset For SVM training and then trained it (Not necessary because the project already contains the .xml of the svm trained)"<<endl;
    cout<<" 3- Test the model on the Test Set"<<endl<<endl;

    int option;
    cout<<"Option choosen: ";
    cin>>option;
    if(option==1)
    {   cout<<"[PHASE 1]: DATASET INITIALIZATION FOR CASCADE CLASSIFIER TRAINING"<<endl;
        // Paths of the directory that contains the training set images and the one that contains the txt with the labels
        string trainingSetPath;
        string labeltxtPath;

        cout<<"Insert the Path of the directory with the training Images: ";
        cin>>trainingSetPath;
        cout<<"Insert the Path of the directory with the .txt files of the labels for training Images: ";
        cin>>labeltxtPath;

        //Complete the Paths
        trainingSetPath=trainingSetPath+"/*.*";
        labeltxtPath=labeltxtPath+"/";


        // Paths of files that we need to create
        string fileposName = "Cascade/boats_prof.info";
        string filenegName = "Cascade/bg.txt";

        // Paths of directories that we need to fill up with images
        string path_pos = "Cascade/pos/";
        string path_neg = "Cascade/neg/";



        //- Create the dataset for the Cascade Classifier
        datsetPreparation->createDatasetForCascadeClassifier(trainingSetPath, labeltxtPath, fileposName, filenegName, path_pos, path_neg);
    }
    else if(option==2)
    {   cout<<"[PHASE 2]: CREATION OF DATASET FOR SVM TRAINING AND TRAINED THE SVM USING HOG DESCRIPTORS"<<endl;
        string trainingSetImagesPath;
        cout<<"Insert the Path of the directory with the training Images: ";
        cin>>trainingSetImagesPath;
        trainingSetImagesPath=trainingSetImagesPath+"/";

        // Cascade Classifier Path
        string boatClassifierPath = "models/cascade1.xml";
        // Path of the Boat.info (easyer to read and process in this case)
        string trainingSetInfoPath = "Cascade/boats.info";
        // Paths where to save the images (negative and positive) for the training of the SVM
        string cartellaPositiveForSVM="SVM/posForSVM_canny/";
        string cartellaNegativeForSVM="SVM/negForSVM_canny/";
        // Define the name of the model creted by the professor
        string pathSavingSVM="models/SVM_professor.xml";

        //- Create the dataset
        datsetPreparation->createDatasetForSVM(trainingSetImagesPath, trainingSetInfoPath, boatClassifierPath,cartellaPositiveForSVM,cartellaNegativeForSVM);
        //- Train the SVM
        datsetPreparation->train_svm_hog(cartellaPositiveForSVM+"*.*",cartellaNegativeForSVM+"*.*",pathSavingSVM);
    }
    else if(option==3)
    {   cout<<"[PHASE 3]: TEST THE FINAL COMBINATION OF MODELS ON TEST SET AND GET THE RESULTS"<<endl;

        string testSetPath;
        string testSetLabelsPath;

        //- Ask the path
        cout<<"Insert the path of the directory with Images of Testset:";
        cin>>testSetPath;
        cout<<endl<<"Insert the path of the directory with .txt Files of Labels of Testset:";
        cin>>testSetLabelsPath;

        //Complete the Paths
        testSetPath=testSetPath+"/";
        testSetLabelsPath=testSetLabelsPath+"/";


        //- Test the final Model on test set
        testTheFinalModel(testSetPath,testSetLabelsPath);
    }

    return 0;
}

void testTheFinalModel(string testSetPath,string testSetLabelsPath)
{
    //- Define which models to use in my Combined Model
    string pathClassifier="models/cascade1.xml";
    string pathClassifier2="models/cascade2.xml";
    string pathSVM="models/SVM_HOG_Canny.xml";


    //- Create an istance of Combined Model
    CombinedModel *model=new CombinedModel(pathClassifier,pathClassifier2,pathSVM);
    //- Get the results on the test set
    vector<BoatImage> imagesResult=model->getResultTestSet(testSetPath,true);

    //- Prepare The images of the Ground Truth
    vector<BoatImage> groundTruthImages;
    for (int i = 0; i < imagesResult.size(); ++i) {
        ifstream labelsFile(testSetLabelsPath + imagesResult[i].getNameNoExtension()+".txt");
        if (labelsFile.is_open()) {
            //- Create a new Boat image
            BoatImage imageTruth=*(new BoatImage(imagesResult[i].getName(),imagesResult[i].getPath()));

            //- Read the Ground Truth from the .txt files
            string str;
            vector<string> lines;
            while (getline(labelsFile, str)) {lines.push_back(str);}

            //- Save the Bounding Boxes into my BoatImage
            vector<vector<string>> values;
            vector<BoundingBox> realBboxes;
            for (int i = 0; i < lines.size(); ++i) {
                //- Each line contains a Bounding Box, so read all the values
                vector<string> comodo;
                comodo = datsetPreparation->split(lines[i], ':');
                vector<string> bboxes = datsetPreparation->split(comodo[1], ';');;
                int xmin = stoi(bboxes[0]);
                int xmax = stoi(bboxes[1]);
                int ymin = stoi(bboxes[2]);
                int ymax = stoi(bboxes[3]);

                //- Create the Bounding Box
                BoundingBox b=*(new BoundingBox(xmin,ymin,xmax-xmin,ymax-ymin,1));
                realBboxes.push_back(b);
            }
            //- Add all the Bounding Boxes to the BoatImage
            imageTruth.addBoundingBoxes(realBboxes);

            //- Add the BoatImage of the Ground truth to a vector (in order to show it then to the user)
            groundTruthImages.push_back(imageTruth);
        }
    }

    //- Show the results to the user
    if(imagesResult.size()>0)
        cout<<endl<<"Images:"<<endl;
    for (int i = 0; i < imagesResult.size() && i<groundTruthImages.size(); ++i) {
        //- Print all the information about the results of the model compared to the ground Truth (IoU, Boxes Found, Boats missing, False Positive)
        cout<<imagesResult[i].getIoU(groundTruthImages[i])<<endl<<endl;

        // Concatenate the resulting image of the model with the ground truth for a better visualization and Show it
        Mat outputImage;
        hconcat(imagesResult[i].getImageWithBoxes(),groundTruthImages[i].getImageWithBoxes(),outputImage);
        resize(outputImage,outputImage,Size(outputImage.cols/2.0,outputImage.rows/2.0));
        imshow("Combined Model Result      |      Ground Truth", outputImage);
        waitKey(0);
    }


}







