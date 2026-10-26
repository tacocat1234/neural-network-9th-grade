#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cmath>
#include "neuralNetwork.hpp"
#include "functions.hpp"
#include "mnist.hpp"

void runXor(){
    trainingData xorData = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}},
    };

    NeuralNetwork network({2, 4, 1}, sigmoid, sigmoidDerivative, 1.0, 4);

    network.train(20000, xorData);

    for (auto& dataPoint : xorData){
        std::vector<double> output = network.evaluate(dataPoint.first);
        std::cout << dataPoint.first[0] << " XOR " << dataPoint.first[1]
                  << " -> " << output[0] << " (expected " << dataPoint.second[0] << ")\n";
    }
}

//helper for % samples where highest output matches labeled digit
double accuracy(NeuralNetwork& network, trainingData& dataSet){
    int correct = 0;
    for (auto& dataPoint : dataSet){
        if (indexMax(network.evaluate(dataPoint.first)) == indexMax(dataPoint.second)){
            correct++;
        }
    }
    return correct * 100.0 / dataSet.size();
}

void runMnist(const std::string& dataDir){
    const int batchSize = 10;
    const int epochs = 5;
    const int testSamples = 10000;
    const int trainSamples = 60000;

    std::cout << "loading MNIST from " << dataDir << "...\n";
    trainingData trainData = loadMnistData(dataDir + "/train-images-idx3-ubyte", dataDir + "/train-labels-idx1-ubyte", trainSamples);
    trainingData testData = loadMnistData(dataDir + "/t10k-images-idx3-ubyte", dataDir + "/t10k-labels-idx1-ubyte", testSamples);
    std::cout << trainData.size() << " training samples, " << testData.size() << " test samples\n";

    //lr 3.0, high lr since data is simple
    NeuralNetwork network({784, 32, 10}, sigmoid, sigmoidDerivative, 3.0, batchSize);

    std::cout << "test accuracy before training: " << accuracy(network, testData) << "%\n";

    int batchesPerEpoch = std::ceil(trainData.size() / static_cast<double>(batchSize));
    for (int epoch = 1; epoch <= epochs; epoch++){
        network.train(batchesPerEpoch, trainData); //one pass over shuffeled training set
        std::cout << "epoch " << epoch << ": test accuracy " << accuracy(network, testData) << "%\n";
    }

    for (int i = 0; i < 10; i++){
        std::cout << "test image " << i << ": predicted " << indexMax(network.evaluate(testData[i].first))
                  << ", actual " << indexMax(testData[i].second) << "\n";
    }
}

void printElapsed(const std::string& name, std::chrono::steady_clock::time_point start){
    std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    std::cout << name << " took " << std::fixed << std::setprecision(1) << seconds.count() << " seconds\n"
              << std::defaultfloat << std::setprecision(6); //restore default formatting
}

int main(int argc, char* argv[]){
    // wait here so I can set up Windows perfmon to capture this process
    // std::cout << "==== Type any key to continue ====\n";
    // char c;
    // std::cin >> c;

    std::cout << "=== XOR ===\n";
    auto start = std::chrono::steady_clock::now();
    runXor();
    printElapsed("runXor", start);

    std::cout << "\n=== MNIST ===\n";
    start = std::chrono::steady_clock::now();
    runMnist(argc > 1 ? argv[1] : "data/mnist"); //optionally pass MNIST folder as the first arg
    printElapsed("runMnist", start);
    return 0;
}
