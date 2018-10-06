//
// Created by alex on 06.10.18.
//

#ifndef INC_01_DATASET_H
#define INC_01_DATASET_H

#include <iostream>

class Dataset
{
public:
    float * getPointFeatures(int i);

    float * getPointLabels(int i);

    int getNPoints();

    Dataset(int nPoints, int nFeatures, int nLabels);

    explicit Dataset(std::istream& is);

    Dataset(void (*func)(float * in, float * out), int nIn, int nOut, float * start, float * end, int parts, bool stepIn = true);

    friend std::ostream& operator<< (std::ostream& os, Dataset * dataset);

    void setPoint(int i, float * features, float * labels);

    int getNFeatures();

    int getNLabels();

    ~Dataset();

private:
    int nPoints;
    int nFeatures;
    int nLabels;
    float ** pointFeatures;
    float ** pointLabels;
};


#endif //INC_01_DATASET_H
