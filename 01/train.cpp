#include <cmath>
#include <ostream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <fstream>

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
			for (int j = 0; j < nFeaturesIn + 1; ++j)
				this->weights[i][j] = (float)random() / (float)RAND_MAX / 10.0 / (float)(nFeaturesIn + 1);
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
		os << layer->nFeaturesOut << " " << layer->nFeaturesIn << std::endl;
        for (int neuron = 0; neuron < layer->nFeaturesOut; ++neuron) {
            for (int feature = 0; feature < layer->nFeaturesIn + 1; ++feature)
                os << layer->getWeights(neuron)[feature] << " ";
			os << std::endl;
        }
        return os;
    }

	void setFeaturesIn(float * featuresIn)
	{
		this->featuresIn = featuresIn;
	}

	Layer(std::istream& is, Layer * previous)
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

	Dataset(std::istream& is)
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

	friend std::ostream& operator<< (std::ostream& os, Dataset * dataset)
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

    void setPoint(int i, float * features, float * labels)
    {
        for (int j = 0; j < nFeatures; ++j) {
            pointFeatures[i][j] = features[j];
        }

        for (int j = 0; j < nLabels; ++j) {
            pointLabels[i][j] = labels[j];
        }
    }

	int getNFeatures()
	{
		return this->nFeatures;
	}

	int getNLabels()
	{
		return this->nLabels;
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
            for (int i = 1; i < nHiddenLayers; ++i) {
                layer = new Layer(layer, nNeurons[i]);
            }
            this->last = new Layer(layer, nFeaturesOut);
        }
    }

	Network(Dataset * trainset, int nHiddenLayers, int nNeurons[]) : Network(trainset->getNFeatures(), trainset->getNLabels(), nHiddenLayers, nNeurons)
	{
		
	}

    Network(std::istream& is)
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

    void trainEpoch(Dataset & dataset)
    {
		float kickLoss = 0.01;
		float finishDeltaLoss = 0.00000001;
		float speed = 50;
		float shuffle = 20;
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

		float L1 = calculateLoss(dataset);
		if (!(L > kickLoss && ((L - L1) < finishDeltaLoss * speed)))
			return;

		for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
            Layer * layer = getLayer(iLayer);
            for (int neuron = 0; neuron < layer->getNFeaturesOut(); ++neuron) {
                for (int feature = 0; feature < layer->getNFeaturesIn() + 1; ++feature) {
                    shuffleWeight(layer, neuron, feature, shuffle, dw);
                }
            }
        }

    }

	void showDistribution(float xMin, float xMax, int nX, float yMin, float yMax, int nY, int sizeX, int sizeY, Dataset * dataset=nullptr)
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

	void printMistakes(std::ostream& os, Dataset * dataset)
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

	void train(Dataset * dataset, float needLoss = 0.01, std::ostream * info = &std::cerr)
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
				showDistribution(-1, 1, 50, -1, 1, 50, 600, 600, dataset);
				cv::waitKey(1);
			}
			
			if (info != nullptr)
				*info << loss << std::endl;

			if (loss < needLoss)
			    break;
	    }
	}

	friend std::ostream& operator << (std::ostream& os, Network * net)
	{
		os << net->nLayers << std::endl;
		for (int i = 0; i < net->nLayers; i++)
			os << net->getLayer(i);
		os << std::endl;
		return os;
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

	float shuffleWeight(Layer * layer, int neuron, int feature, float percent, float delta)
    {
        layer->getWeights(neuron)[feature] *= (1 + percent / 100 * (random() / RAND_MAX - 0.5) * 2);
		layer->getWeights(neuron)[feature] += delta * (random() / RAND_MAX - 0.5) * 2;
    }

    float loss(float * labelsCalculated, float * labelsDataset, int nLabels)
    {
        float L = 0;
        for (int i = 0; i < nLabels; ++i) {
            float delta = labelsCalculated[i] - labelsDataset[i];
            L += delta * delta;
        }
        return L / nLabels;
    }
};

int main()
{
	std::ifstream datafile("../datasets/testsetBig.txt");
	Dataset * dataset = new Dataset(datafile);
	datafile.close();

	std::ifstream netfile("netDigits3.txt");
	Network * net = new Network(netfile);
	netfile.close();

/*
	Network * net = new Network(dataset, 0, new int[2]{2,7});
	net->train(dataset, 0.05);
 
	std::ofstream netSaveFile;
	netSaveFile.open("net.txt");
	netSaveFile << net;
	netSaveFile.close();

	std::cout << "------------------------\n";
*/

	net->printMistakes(std::cout, dataset);

	//cv::waitKey(0);

    return 0;
}
