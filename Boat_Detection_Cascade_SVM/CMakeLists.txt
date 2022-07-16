cmake_minimum_required(VERSION 2.8)
project(ProjectCV)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 14)

add_executable(ProjectCV project.cpp BoundingBox.cpp BoundingBox.h BoatImage.cpp BoatImage.h CombinedModel.cpp CombinedModel.h datasetPreparation_utils.cpp datasetPreparation_utils.h)

target_link_libraries( ProjectCV ${OpenCV_LIBS} )