//
// Created by alex on 06.10.18.
//

#ifndef INC_01_LAYER_H
#define INC_01_LAYER_H

#include <iostream>

class Layer
{
public:
    Layer(float* featuresIn, int nFeaturesIn, int nFeaturesOut);

    Layer(Layer * previous, int nFeaturesOut);

    void process();

    void processAll();

    float * getFeaturesOut();

    int getNFeaturesOut();

    Layer *getPrevious();

    float * getWeights(int neuron);

    int getNFeaturesIn();

    friend std::ostream& operator << (std::ostream& os, Layer * layer);

    void setFeaturesIn(float * featuresIn);

    Layer(std::istream& is, Layer * previous);

    ~Layer();

private:
    int nFeaturesIn;
    int nFeaturesOut;
    float ** weights;
    float * featuresIn;
    float * featuresOut;

    Layer * previous;

    float outClampFunc(float y);
};


#endif //INC_01_LAYER_H
