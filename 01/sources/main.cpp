//
// Created by alex on 06.10.18.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Dataset.h"
#include "Network.h"

#define N_SAMPLES 20

float f1(float x)
{
    return 2.0f / (2.0f + x * x);
}

float f2(float x)
{
    return 5.0f / (5.0f + x * x);
}

void func(float * in, float * out)
{
    for (int i = 0; i < N_SAMPLES; ++i) {
        in[i] = (out[0] - 0.5f) * f1((float)i / N_SAMPLES) + (out[1] - 0.5f) * f2((float)i / N_SAMPLES);
    }
}

int main()
{
    /*
    std::ifstream datafile("dataset.txt");
    Dataset * dataset = new Dataset(datafile);
    datafile.close();
    */

    Dataset * dataset = new Dataset(func, N_SAMPLES, 2, new float[2] {0.01f, 0.01f}, new float[2] {0.99f, 0.99f}, 20, false);

/*
	std::ifstream netfile("net.txt");
	Network * net = new Network(netfile);
	netfile.close();
*/


    srandom(time(NULL));
    Network * net = new Network(dataset, 0, new int[2]{2,4});
    net->train(dataset, 0.001);

    std::ofstream netSaveFile;
    netSaveFile.open("net.txt");
    netSaveFile << net;
    netSaveFile.close();

    std::cout << "------------------------\n";


    float * signal = new float[40];
    float * spectrum = new float[2] {0.35f, 0.45f};

    func(signal, spectrum);

    net->copyFeaturesIn(signal);
    net->process();
    float * spectrumOut = net->getFeaturesOut();

    std::cout << spectrumOut[0] << "\n" << spectrumOut[1] << "\n";

//	net->printMistakes(std::cout, dataset);

    cv::waitKey(0);

    return 0;
}
