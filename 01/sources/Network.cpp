//
// Created by alex on 06.10.18.
//

#include "Network.h"
#include <opencv2/opencv.hpp>


Network::Network(int nFeaturesIn, int nFeaturesOut, int nHiddenLayers, int nNeurons[])
{
    this->nLayers = nHiddenLayers + 1;
    this->nFeatures = nFeaturesIn;
    this->features = new float[nFeatures];

    if (nHiddenLayers == 0)
        last = new Layer(features, nFeatures, nFeaturesOut);
    else
    {
        Layer * layer = new Layer(features, nFeatures, nNeurons[0]);
        for (int i = 1; i < nHiddenLayers; ++i) {
            layer = new Layer(layer, nNeurons[i]);
        }
        this->last = new Layer(layer, nFeaturesOut);
    }
}

Network::Network(Dataset * trainset, int nHiddenLayers, int nNeurons[]) :
        Network(trainset->getNFeatures(), trainset->getNLabels(), nHiddenLayers, nNeurons) {}

Network::Network(std::istream& is)
{
    is >> nLayers;
    last = nullptr;
    for (int i = 0; i < nLayers; i++)
        last = new Layer(is, last);
    Layer * first = this->getLayer(0);
    this->nFeatures = first->getNFeaturesIn();
    this->features = new float[nFeatures];
    first->setFeaturesIn(features);
}

void Network::copyFeaturesIn(float * features)
{
    for (int i = 0; i < nFeatures; ++i) {
        this->features[i] = features[i];
    }
}

float * Network::getFeaturesOut()
{
    return last->getFeaturesOut();
}

void Network::process()
{
    last->processAll();
}

int Network::getNFeaturesIn()
{
    return nFeatures;
}

int Network::getNFeaturesOut()
{
    return last->getNFeaturesOut();
}

float Network::calculateLoss(Dataset & dataset)
{
    float L = 0;
    for (int i = 0; i < dataset.getNPoints(); ++i) {
        copyFeaturesIn(dataset.getPointFeatures(i));
        process();
        L += loss(getFeaturesOut(), dataset.getPointLabels(i), getNFeaturesOut());
    }
    return L / dataset.getNPoints();
}

Layer * Network::getLayer(int i)
{
    if (i < 0)
        return nullptr;
    Layer * last = this->last;
    int n = nLayers;

    while (i < n-1 && last != nullptr) {
        last = last->getPrevious();
        n--;
    }
    return last;
}

void Network::trainEpoch(Dataset & dataset)
{
    //float kickLoss = 0.10;
    //float finishDeltaLoss = 0.00001;
    float speed = 10;
    //float shuffle = 20;
    float dw = 0.1;
    float maxDw = 0.1;


    float L = calculateLoss(dataset);
    for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
        Layer * layer = getLayer(iLayer);
        for (int neuron = 0; neuron < layer->getNFeaturesOut(); ++neuron) {
            for (int feature = 0; feature < layer->getNFeaturesIn() + 1; ++feature) {
                shiftWeight(layer, neuron, feature, dw);
                process();
                float dL = calculateLoss(dataset) - L;
                float Dw = - speed * dL / dw;
                if (std::abs(Dw) > maxDw)
                    Dw = Dw / std::abs(Dw) * maxDw;
                //std::cout << Dw << "\n";

                shiftWeight(layer, neuron, feature, -dw + Dw);
            }
        }
    }

    /*
    float L1 = calculateLoss(dataset);
    if (!(L > kickLoss && (std::abs(L - L1) < finishDeltaLoss * speed)))
        return;

    for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
        Layer * layer = getLayer(iLayer);
        for (int neuron = 0; neuron < layer->getNFeaturesOut(); ++neuron) {
            for (int feature = 0; feature < layer->getNFeaturesIn() + 1; ++feature) {
                shuffleWeight(layer, neuron, feature, shuffle, dw);
            }
        }
    }
     */

}

void Network::showDistribution(float xMin, float xMax, int nX, float yMin, float yMax, int nY, int sizeX, int sizeY, Dataset * dataset)
{
    if (this->getNFeaturesIn() != 2 || this->getNFeaturesOut() != 1)
        return;

    cv::Mat M0(nY, nX, CV_8UC3);
    for (int i = 0; i < nY; i++)
        for (int j = 0; j < nX; j++)
        {
            float x = xMin + j * (xMax - xMin) / nX;
            float y = yMin + (nY - i) * (yMax - yMin) / nY;
            float xx[2] = {x, y};
            this->copyFeaturesIn(xx);
            this->process();
            float p = this->getFeaturesOut()[0];
            M0.at<cv::Vec3b>(i, j) = cv::Vec3b::all(p * 255);
        }

    cv::Mat M(sizeY, sizeX, CV_8UC3);
    cv::resize(M0, M, cv::Size2i(sizeX, sizeY), 0, 0, cv::INTER_NEAREST);

    if (dataset == nullptr)
        goto draw;

    for (int point = 0; point < dataset->getNPoints(); point++)
    {
        float x = dataset->getPointFeatures(point)[0];
        float y = dataset->getPointFeatures(point)[1];
        int j =    (x - xMin) / (xMax - xMin)  * sizeX;
        int i = (1-(y - yMin) / (yMax - yMin)) * sizeY;
        const cv::Vec3b color = dataset->getPointLabels(point)[0] > 0.5f ? cv::Vec3b(0,0,255) : cv::Vec3b(255,0,0);
        cv::circle(M, cv::Point(j, i), 5, color, 2);
    }

    draw:

    cv::imshow("Distribution", M);
}

void Network::printMistakes(std::ostream& os, Dataset * dataset)
{
    os << "Mistakes (ids): \n\n";
    int n = dataset->getNPoints();
    int mistaken = 0;
    for (int i = 0; i < n; ++i) {
        copyFeaturesIn(dataset->getPointFeatures(i));
        process();
        float L = loss(getFeaturesOut(), dataset->getPointLabels(i), getNFeaturesOut());
        if (L > 0.5)
        {
            os << i << "\twith loss " << L << std::endl;
            mistaken++;
        }
    }
    os << "\nTotal mistakes: " << mistaken << " out of " << n << "(" << mistaken * 100 / n << "%)" << std::endl;
}

void Network::train(Dataset * dataset, float needLoss, std::ostream * info)
{
    if (this->nFeatures == 2 && this->getNFeaturesOut() == 1)
        cv::namedWindow("Distribution");

    float loss;
    int i = 0;
    for (;1; ++i) {
        trainEpoch(*dataset);

        loss = calculateLoss(*dataset);

        if (this->nFeatures == 2 && this->getNFeaturesOut() == 1)
        {
            showDistribution(-1.2, 1.2, 50, -1.2, 1.2, 50, 600, 600, dataset);
            cv::waitKey(1);
        }

        if (info != nullptr)
            *info << loss << std::endl;

        if (loss < needLoss)
            break;
    }
}

std::ostream& operator << (std::ostream& os, Network * net)
{
    os << net->nLayers << std::endl;
    for (int i = 0; i < net->nLayers; i++)
        os << net->getLayer(i);
    os << std::endl;
    return os;
}

Network::~Network()
{
    delete last;
    delete features;
}


float Network::shiftWeight(Layer * layer, int neuron, int feature, float delta)
{
    layer->getWeights(neuron)[feature] += delta;
}

float Network::shuffleWeight(Layer * layer, int neuron, int feature, float percent, float delta)
{
    layer->getWeights(neuron)[feature] *= (1 + percent / 100 * (random() / RAND_MAX - 0.5) * 2);
    layer->getWeights(neuron)[feature] += delta * (random() / RAND_MAX - 0.5) * 2;
}

float Network::loss(float * labelsCalculated, float * labelsDataset, int nLabels) {
    float L = 0;
    for (int i = 0; i < nLabels; ++i) {
        float delta = labelsCalculated[i] - labelsDataset[i];
        L += delta * delta;
    }
    return L / nLabels;
}