//
// Created by luca on 10/07/21.
//

#include "BoundingBox.h"


//- CONSTRUCTORS
BoundingBox::BoundingBox(int x, int y, int width, int height) {
    this->x=x;
    this->y=y;
    this->width=width;
    this->height=height;
    goodness=0;
}
BoundingBox::BoundingBox(int x, int y, int width, int height, float goodness) {
    this->x=x;
    this->y=y;
    this->width=width;
    this->height=height;
    this->goodness=goodness;
}
BoundingBox::BoundingBox(Rect rect) {
    this->x=rect.x;
    this->y=rect.y;
    this->width=rect.width;
    this->height=rect.height;
    goodness=0;
}
BoundingBox::BoundingBox(Rect rect, float goodness_) {
    this->x=rect.x;
    this->y=rect.y;
    this->width=rect.width;
    this->height=rect.height;
    goodness=goodness_;
}

// GETTER
int BoundingBox::getX() {
    return x;
}

int BoundingBox::getY() {
    return y;
}

int BoundingBox::getHeight() {
    return height;
}

int BoundingBox::getWidth() {
    return width;
}

float BoundingBox::getGoodness() {
    return goodness;
}

vector<Point> BoundingBox::getVerteces() {
    //Return the vertices (Points) of the Bounding Box

    Point p1,p2,p3,p4;
    p1.x=x;
    p1.y=y;

    p2.x=x+width;
    p2.y=y;

    p3.x=x;
    p3.y=y+height;

    p4.x=x+width;
    p4.y=y+height;

    verteces.push_back(p1);
    verteces.push_back(p2);
    verteces.push_back(p3);
    verteces.push_back(p4);
    return verteces;
}

Rect BoundingBox::getRect() {
    Rect r=*(new Rect(x,y,width,height));
    return r;
}

String BoundingBox::getInfo() {
    // get a string with all the information of the Bounding Box
    string s="{ X: "+to_string(x)+"   Y: "+to_string(y)+"   Width: "+to_string(width)+"   Height: "+to_string(height)+" }";
    return s;
}

float BoundingBox::IntersectionOverUnion(BoundingBox b) {
    // Calculate the value of Intersection Over Union of this Bounding Box with another one (b)
    float interArea=this->getBoxOfIntersection(b).getArea();
    if(interArea==0) return 0;
    float iou=interArea/(this->getArea()+b.getArea()-interArea);
    return iou;
}

float BoundingBox::getArea() {
    // return the value of the Area of the Bounding Box
    return width*height;
}

BoundingBox BoundingBox::getBoxOfIntersection(BoundingBox b) {
    // return the Bounding box create as the intersection of this and b
    if(this->hasIntersectionWith(b))
    {
        int x_i=max(this->x,b.getX());
        int y_i=max(this->y,b.getY());
        int x2_i=min(this->x+ this->width,b.getX()+b.getWidth());
        int y2_i=min(this->y+ this->height,b.getY()+b.getHeight());
        return BoundingBox(x_i, y_i, x2_i-x_i, y2_i-y_i);
    }
    else
        return BoundingBox(0, 0, 0, 0);
}

BoundingBox BoundingBox::getBoxAroundUnion(BoundingBox b) {

    // Create a Bounding Box that contains completely this Bounding Box and the Bounding Box b
    int x_=min(this->x,b.getX());
    int y_=min(this->y,b.getY());
    int height_=this->height+b.getHeight()-this->getBoxOfIntersection(b).getHeight();
    int width_=this->width+b.getWidth()-this->getBoxOfIntersection(b).getWidth();

    return BoundingBox(x_, y_,width_, height_);
}


// SETTER
void BoundingBox::setGoodness(float value) {
    goodness=value;
}

// COMPARISONS
bool BoundingBox::hasIntersectionWith(BoundingBox b) {

    //Returns true if two rectangles (l1, r1) and (l2, r2)  overlap
    vector<Point> vertecesB=b.getVerteces();
    // If one rectangle is on left side of other
    if (x >= vertecesB[1].x || vertecesB[0].x >= (x+width))
        return false;

    // If one rectangle is above other
    if (y >= vertecesB[2].y || vertecesB[0].y >= (y+height))
        return false;

    return true;
}

bool BoundingBox::isContainedIn(BoundingBox b) {
    //- Check if another Bounding Box b is completely contained in this
    if((x>=b.getX())&&(y>=b.getY())&&((x+width)<=(b.getX()+b.getWidth()) )&& ((y+height)<=(b.getY()+b.getHeight()) )  )
        return true;
    else
        return false;
}










