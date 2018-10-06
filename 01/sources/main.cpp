//
// Created by alex on 06.10.18.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Dataset.h"
#include "Network.h"

void func(float * in, float * out)
{
    out[0] = (in[0] * in[1] > 0.07) ? 1 : 0;
}

int main()
{
    /*
    std::ifstream datafile("dataset.txt");
    Dataset * dataset = new Dataset(datafile);
    datafile.close();
    */

    Dataset * dataset = new Dataset(func, 2, 1, new float[2] {-1, -1}, new float[2] {1, 1}, 20);

    std::cout << dataset;

    /*
	std::ifstream netfile("netDigits3.txt");
	Network * net = new Network(netfile);
	netfile.close();

*/
    srandom(time(NULL));
    Network * net = new Network(dataset, 1, new int[2]{16,1});
    net->train(dataset, 0.01);

    std::ofstream netSaveFile;
    netSaveFile.open("net.txt");
    netSaveFile << net;
    netSaveFile.close();

    std::cout << "------------------------\n";

//	net->printMistakes(std::cout, dataset);

    cv::waitKey(0);

    return 0;
}
