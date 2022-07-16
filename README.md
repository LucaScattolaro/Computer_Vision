# Computer_Vision

## BOAT DETECTION FINAL PROJECT (no deep learning)
The goal of this final project is to develop a system capable of detecting boats in the images. Automatic boat
detection plays an important role in different fields for example maritime surveillance.
The general Idea of my solution is to use Cascade of Classifiers in order to detect all the boats and then process
the results obtained using HOG descriptors and SVM in order to discard Non-Boat Bounding Boxes and use
as post processing, to get better results if needed, the Canny edge detector.
In this report I will describe the reasons of the choices that I have done for my final model and at the end I
will show qualitative and quantitative results on the test set.


## LAB 2 - Camera Calibration
Camera calibration can be defined as the technique of estimating the characteristics of a camera. 
It means that we have all of the camera’s information like parameters or coefficients which are needed to determine an accurate relationship 
between a 3D point in the real world and its corresponding 2D projection in the image acquired by that calibrated camera.


## LAB 3
This homework is composed by 2 parts:
<ol>
  <li> Histogram Equalization</li>
  <li>  Image Filtering
  <ul>
      <li>Median Filter</li>
      <li>Gaussian Filter</li>
      <li>Bilateral Filter</li>
    </ul>
</li>
  
</ol>

## LAB 5
Computational Steps
<ol>
  <li> Load a set of images</li>
  <li> Project the images on a cylinder surface</li>
  <li> Compute the Panorama
    <ul>
      <li>Extract the SIFT features from the images</li>
      <li> Compute and refine the match between images</li>
      <li>Compute the final panorama</li>
    </ul>
  </li>
</ol>
