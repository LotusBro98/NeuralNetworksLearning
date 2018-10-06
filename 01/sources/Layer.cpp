//
// Created by alex on 06.10.18.
//

#include <cmath>
#include "Layer.h"

Layer::Layer(float* featuresIn, int nFeaturesIn, int nFeaturesOut)
{
    this->previous = nullptr;
    this->nFeaturesIn = nFeaturesIn;
    this->nFeaturesOut = nFeaturesOut;
    this->featuresIn = featuresIn;

    this->featuresOut = new float[nFeaturesOut];
    this->weights = new float*[nFeaturesOut];
    for (int i = 0; i < nFeaturesOut; ++i) {
        this->weights[i] = new float[nFeaturesIn + 1];
        for (int j = 0; j < nFeaturesIn + 1; ++j)
            this->weights[i][j] = (float)random() / (float)RAND_MAX / 10.0f / (float)(nFeaturesIn + 1);
    }
}

Layer::Layer(Layer * previous, int nFeaturesOut) : Layer(previous->featuresOut, previous->nFeaturesOut, nFeaturesOut) {
    this->previous = previous;
}

void Layer::process()
{
    for (int neuron = 0; neuron < nFeaturesOut; ++neuron) {
        float y = weights[neuron][nFeaturesIn];
        for (int feature = 0; feature < nFeaturesIn; ++feature) {
            y += weights[neuron][feature] * featuresIn[feature];
        }
        featuresOut[neuron] = outClampFunc(y);
    }
}

void Layer::processAll()
{
    if (previous != nullptr)
        previous->processAll();
    process();
}

float * Layer::getFeaturesOut()
{
    return featuresOut;
}

int Layer::getNFeaturesOut() {
    return nFeaturesOut;
}

Layer * Layer::getPrevious() {
    return previous;
}

float * Layer::getWeights(int neuron)
{
    return weights[neuron];
}

int Layer::getNFeaturesIn() {
    return nFeaturesIn;
}

std::ostream& operator << (std::ostream& os, Layer * layer)
{
    os << layer->nFeaturesOut << " " << layer->nFeaturesIn << std::endl;
    for (int neuron = 0; neuron < layer->nFeaturesOut; ++neuron) {
        for (int feature = 0; feature < layer->nFeaturesIn + 1; ++feature)
            os << layer->getWeights(neuron)[feature] << " ";
        os << std::endl;
    }
    return os;
}

void Layer::setFeaturesIn(float * featuresIn)
{
    this->featuresIn = featuresIn;
}

Layer::Layer(std::istream& is, Layer * previous)
{
    is >> nFeaturesOut >> nFeaturesIn;
    weights = new float*[nFeaturesOut];
    for (int i = 0; i < nFeaturesOut; i++)
    {
        weights[i] = new float[nFeaturesIn + 1];
        for (int j = 0; j < nFeaturesIn + 1; j++)
            is >> weights[i][j];
    }

    this->previous = previous;
    if (previous != nullptr)
        this->featuresIn = previous->featuresOut;
    else
        this->featuresIn = nullptr;
    this->featuresOut = new float[nFeaturesOut];
}

Layer::~Layer()
{
    delete[](weights);
    delete featuresOut;
    delete previous;
}

float Layer::outClampFunc(float y)
{
    // sigmoid
    return static_cast<float> (1.0 / (1.0 + std::exp(-y)));
}