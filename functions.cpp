#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include "functions.hpp"

/*
double activationFunction(double stimulus){
    return tanh(stimulus);
}
*/

double tanhDerivative(double stimulus){
    return 1 - tanh(stimulus)*tanh(stimulus);
}

double sigmoid(double stimulus){
    return 1.0/(1.0 + exp(-stimulus));
}

double sigmoidDerivative(double stimulus){
    return sigmoid(stimulus) * (1 - sigmoid(stimulus));
}

double relu(double stimulus){
    return std::max(0.0, stimulus);
}

double reluDerivative(double stimulus){
    return stimulus > 0 ? 1:0;
}

std::vector<trainingData> formMiniBatches(int batchSize, trainingData& dataSet){
    
    //shuffle code
       unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(dataSet.begin(), dataSet.end(), std::default_random_engine(seed));
    
    std::vector<trainingData> output;
    output.resize(ceil(dataSet.size()/static_cast<double>(batchSize)));
    
    int i = 0;//i iterates through all points in dataset
    int o = 0;//o iteates through all points in output dataSets
    while (i < dataSet.size()){
        for (int j = 0; j < batchSize && i < dataSet.size(); j++){//j iterates through 0-batchSize
            output[o].push_back(dataSet[i]);
            i++;
        }
        o++;
    }
    return output;
} 

double meanSquareError(std::vector<double> output, std::vector<double> expected){
    double sum = 0;
    for (int i = 0; i < output.size(); i++){
        sum += (output[i] - expected[i])*(output[i] - expected[i]);
    }
    return sum/output.size();
}

double meanSquareDerivative(std::vector<double> output, std::vector<double> expected, int indexOf){
    return 2.0 * (output[indexOf] - expected[indexOf]) / output.size();
}