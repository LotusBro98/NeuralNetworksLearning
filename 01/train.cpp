#include <cmath>
#include <ostream>
#include <iomanip>
#include <iostream>

class Layer
{
public:
    Layer(float* featuresIn, int nFeaturesIn, int nFeaturesOut)
    {
        this->previous = nullptr;
        this->nFeaturesIn = nFeaturesIn;
        this->nFeaturesOut = nFeaturesOut;
        this->featuresIn = featuresIn;

        this->featuresOut = new float[nFeaturesOut];
        this->weights = new float*[nFeaturesOut];
        for (int i = 0; i < nFeaturesOut; ++i) {
            this->weights[i] = new float[nFeaturesIn + 1];
        }
    }

    Layer(Layer * previous, int nFeaturesOut) : Layer(previous->featuresOut, previous->nFeaturesOut, nFeaturesOut) {
        this->previous = previous;
    }

    void process()
    {
        for (int neuron = 0; neuron < nFeaturesOut; ++neuron) {
            float y = weights[neuron][nFeaturesIn];
            for (int feature = 0; feature < nFeaturesIn; ++feature) {
                y += weights[neuron][feature] * featuresIn[feature];
            }
            featuresOut[neuron] = outClampFunc(y);
        }
    }

    void processAll()
    {
        if (previous != NULL)
            previous->processAll();
        process();
    }

    float * getFeaturesOut()
    {
        return featuresOut;
    }

    int getNFeaturesOut() {
        return nFeaturesOut;
    }

    Layer *getPrevious() const {
        return previous;
    }

    float * getWeights(int neuron)
    {
        return weights[neuron];
    }

    int getNFeaturesIn() const {
        return nFeaturesIn;
    }

    friend std::ostream& operator << (std::ostream& os, Layer * layer)
    {
        os << "------------------------------\n";
        for (int neuron = 0; neuron < layer->nFeaturesOut; ++neuron) {
            os << '(';
            for (int feature = 0; feature < layer->nFeaturesIn + 1; ++feature) {
                os << std::setw(4) << std::setprecision(2) << layer->getWeights(neuron)[feature];
                if (feature == layer->nFeaturesIn)
                    os << ")\n";
                else
                    os << ", ";
            }
        }
        os << "------------------------------\n";
        return os;
    }

    ~Layer()
    {
        delete[](weights);
        delete featuresOut;
        delete previous;
    }

private:
    int nFeaturesIn;
    int nFeaturesOut;
    float ** weights;
    float * featuresIn;
    float * featuresOut;

    Layer * previous;

    float outClampFunc(float y)
    {
        // sigmoid
        return static_cast<float> (1.0 / (1.0 + std::exp(-y)));
    }
};

class Dataset
{
public:
    float * getPointFeatures(int i)
    {
        return pointFeatures[i];
    }

    float * getPointLabels(int i)
    {
        return pointLabels[i];
    }

    int getNPoints()
    {
        return nPoints;
    }

    Dataset(int nPoints, int nFeatures, int nLabels)
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

    void setPoint(int i, float * features, float * labels)
    {
        for (int j = 0; j < nFeatures; ++j) {
            pointFeatures[i][j] = features[j];
        }

        for (int j = 0; j < nLabels; ++j) {
            pointLabels[i][j] = labels[j];
        }
    }

    ~Dataset()
    {
        delete [] pointFeatures;
        delete [] pointLabels;
    }

private:
    int nPoints;
    int nFeatures;
    int nLabels;
    float ** pointFeatures;
    float ** pointLabels;
};

class Network
{
public:
    Network(int nFeaturesIn, int nFeaturesOut, int nHiddenLayers, int nNeurons[])
    {
        this->nLayers = nHiddenLayers + 1;
        this->nFeatures = nFeaturesIn;
        this->features = new float[nFeatures];

        if (nHiddenLayers == 0)
            last = new Layer(features, nFeatures, nFeaturesOut);
        else
        {
            Layer * layer = new Layer(features, nFeatures, nNeurons[0]);
            for (int i = 1; i < nLayers; ++i) {
                layer = new Layer(layer, nNeurons[i]);
            }
            this->last = layer;
        }
    }

    void copyFeaturesIn(float * features)
    {
        for (int i = 0; i < nFeatures; ++i) {
            this->features[i] = features[i];
        }
    }

    float * getFeaturesOut()
    {
        return last->getFeaturesOut();
    }

    void process()
    {
        last->processAll();
    }

    int getNFeaturesIn()
    {
        return nFeatures;
    }

    int getNFeaturesOut()
    {
        return last->getNFeaturesOut();
    }

    float calculateLoss(Dataset & dataset)
    {
        float L = 0;
        for (int i = 0; i < dataset.getNPoints(); ++i) {
            copyFeaturesIn(dataset.getPointFeatures(i));
            process();
            L += loss(getFeaturesOut(), dataset.getPointLabels(i), getNFeaturesOut());
        }
        return L / dataset.getNPoints();
    }

    Layer * getLayer(int i)
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

    void trainEpoch(Dataset & dataset, float speed, float dw)
    {
        float L = calculateLoss(dataset);
        for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
            Layer * layer = getLayer(iLayer);
            for (int neuron = 0; neuron < layer->getNFeaturesOut(); ++neuron) {
                for (int feature = 0; feature < layer->getNFeaturesIn() + 1; ++feature) {
                    shiftWeight(layer, neuron, feature, dw);
                    process();
                    float dL = calculateLoss(dataset) - L;
                    shiftWeight(layer, neuron, feature, -dw - speed * dL / dw);
                }
            }
        }
    }

    ~Network()
    {
        delete last;
        delete features;
    }

private:
    int nLayers;
    Layer * last;
    float * features;
    int nFeatures;

    float shiftWeight(Layer * layer, int neuron, int feature, float delta)
    {
        layer->getWeights(neuron)[feature] += delta;
    }

    float loss(float * labelsCalculated, float * labelsDataset, int nLabels)
    {
        float L = 0;
        for (int i = 0; i < nLabels; ++i) {
            float delta = labelsCalculated[i] - labelsDataset[i];
            L += std::abs(delta);
        }
        return L / nLabels;
    }
};

int main()
{
    Dataset * dataset = new Dataset(10, 2, 1);
    dataset->setPoint(0, new float[2]{0, 1}, new float[2]{1});
    dataset->setPoint(1, new float[2]{1, 2}, new float[2]{1});
    dataset->setPoint(2, new float[2]{2, 3}, new float[2]{1});
    dataset->setPoint(3, new float[2]{3, 4}, new float[2]{1});
    dataset->setPoint(4, new float[2]{4, 5}, new float[2]{1});
    dataset->setPoint(5, new float[2]{5, 8}, new float[2]{0});
    dataset->setPoint(6, new float[2]{6, 9}, new float[2]{0});
    dataset->setPoint(7, new float[2]{7, 10}, new float[2]{0});
    dataset->setPoint(8, new float[2]{8, 11 }, new float[2]{0});
    dataset->setPoint(9, new float[2]{9, 12 }, new float[2]{0});

    Network * net = new Network(2, 1, 0, new int[0]{});

    float loss = 1;
    int i = 0;
    for (; i < 100000; ++i) {
        net->trainEpoch(*dataset, 0.1, 0.1);
        //std::cout << net->getLayer(0);

        loss = net->calculateLoss(*dataset);
        if (loss < 0.001)
            break;
    }

    std::cout << loss << " " << i << "\n";
    std::cout << net->getLayer(0);



    return 0;
}