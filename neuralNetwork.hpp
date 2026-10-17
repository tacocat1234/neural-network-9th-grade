#pragma once

#include <vector>
#include "using.hpp"
#include "nodes.hpp"

class NeuralNetwork{
    public:
        NeuralNetwork(std::vector<double> layerSizes, functionPtr activationFunction, functionPtr activationDerivative, double learningRate, int batchSize); //constructor
        ~NeuralNetwork(); //destructor
        
        void passInputs(std::vector<double> inputs);
        void updateAll();
        std::vector<double> outputValues();
        std::vector<double> evaluate(std::vector<double> inputs);
        void updateWeightsAndBiases();
        void backpropagate(trainingData& dataSample);
        void train(int iterations, trainingData& dataSet);
        networkWeights weights; //weights, 
        std::vector<std::vector<Node*>> nodes; //nodes
        
        std::vector<std::vector<double>> nodeValues;
        std::vector<std::vector<double>> nodeGradients;
        std::vector<std::vector<double>> biasGradients;
        networkWeights weightGradients;
        std::vector<double> layerSizes;
        double learningRate;
        int batchSize;
    private:
        functionPtr activationFunction;
        functionPtr activationDerivative;
};