//
// Created by alex on 06.10.18.
//

#include "Dataset.h"

#include <iostream>
#include <cmath>

float * Dataset::getPointFeatures(int i)
{
    return pointFeatures[i];
}

float * Dataset::getPointLabels(int i)
{
    return pointLabels[i];
}

int Dataset::getNPoints()
{
    return nPoints;
}

Dataset::Dataset(int nPoints, int nFeatures, int nLabels)
{
    this->nPoints  = nPoints;
    this->nFeatures = nFeatures;
    this->nLabels = nLabels;

    this->pointFeatures = new float*[nPoints];
    this->pointLabels = new float*[nPoints];
    for (int i = 0; i < nPoints; ++i) {
        this->pointFeatures[i] = new float[nFeatures];
        this->pointLabels[i] = new float[nLabels];
    }
}

Dataset::Dataset(std::istream& is)
{
    is >> nPoints;
    is >> nFeatures;
    is >> nLabels;

    this->pointFeatures = new float*[nPoints];
    this->pointLabels = new float*[nPoints];
    for (int i = 0; i < nPoints; ++i) {
        this->pointFeatures[i] = new float[nFeatures];
        this->pointLabels[i] = new float[nLabels];
        for (int j = 0; j < nFeatures; j++)
            is >> pointFeatures[i][j];
        for (int j = 0; j < nLabels; j++)
            is >> pointLabels[i][j];
    }
}

std::ostream& operator<< (std::ostream& os, Dataset * dataset)
{
    os << dataset->nPoints << " " << dataset->nFeatures << " " << dataset->nLabels << std::endl;

    for (int point = 0; point < dataset->nPoints; point++)
    {
        for (int i = 0; i < dataset->nFeatures; i++)
            os << dataset->pointFeatures[point][i] << " ";
        for (int i = 0; i < dataset->nLabels; i++)
            os << dataset->pointLabels[point][i] << " ";
        os << std::endl;
    }

    return os;
}

void Dataset::setPoint(int i, float * features, float * labels)
{
    if (features != nullptr)
        for (int j = 0; j < nFeatures; ++j) {
            pointFeatures[i][j] = features[j];
        }

    if (labels != nullptr)
        for (int j = 0; j < nLabels; ++j) {
            pointLabels[i][j] = labels[j];
        }
}

int Dataset::getNFeatures()
{
    return this->nFeatures;
}

int Dataset::getNLabels()
{
    return this->nLabels;
}

Dataset::~Dataset()
{
    delete [] pointFeatures;
    delete [] pointLabels;
}

Dataset::Dataset(void (*func)(float *, float *), int nIn, int nOut, float *start, float *end, int parts)
:Dataset(static_cast<int>(std::pow(parts, nIn)), nIn, nOut)
{
    //float * pointIn = new float[nIn];
    //float * pointOut = new float[nOut];
    for (int i = 0; i < nPoints; ++i) {
        float * pointIn = getPointFeatures(i);
        float * pointOut = getPointLabels(i);

        int num = i;
        for (int j = 0; j < nIn; ++j) {
            pointIn[j] = (num % parts) * (end[j] - start[j]) / parts + start[j];
            num /= parts;
        }

        func(pointIn, pointOut);
    }
}
