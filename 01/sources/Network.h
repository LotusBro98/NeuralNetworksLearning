//
// Created by alex on 06.10.18.
//

#ifndef INC_01_NETWORK_H
#define INC_01_NETWORK_H

#include <iostream>

#include "Dataset.h"
#include "Layer.h"

class Network
{
public:

    Network(int nFeaturesIn, int nFeaturesOut, int nHiddenLayers, int nNeurons[]);

    Network(Dataset * trainset, int nHiddenLayers, int nNeurons[]);

    Network(std::istream& is);

    void copyFeaturesIn(float * features);

    float * getFeaturesOut();

    void process();

    int getNFeaturesIn();

    int getNFeaturesOut();

    float calculateLoss(Dataset & dataset);

    Layer * getLayer(int i);

    void trainEpoch(Dataset & dataset);

    void showDistribution(float xMin, float xMax, int nX, float yMin, float yMax, int nY, int sizeX, int sizeY, Dataset * dataset = nullptr);

    void printMistakes(std::ostream& os, Dataset * dataset);

    void train(Dataset * dataset, float needLoss = 0.01, std::ostream * info = &std::cerr);

    friend std::ostream& operator << (std::ostream& os, Network * net);

    ~Network();

private:

    int nLayers;
    Layer * last;
    float * features;
    int nFeatures;

    float shiftWeight(Layer * layer, int neuron, int feature, float delta);

    float shuffleWeight(Layer * layer, int neuron, int feature, float percent, float delta);

    float loss(float * labelsCalculated, float * labelsDataset, int nLabels);
};


#endif //INC_01_NETWORK_H
