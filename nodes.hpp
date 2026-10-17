#include <vector>
#include "functions.hpp"
#include "neuralNetwork.hpp"


class Node{
    public:
        Node(NeuralNetwork* network, std::pair<int, int> location, double value, double bias);
        
        void updateValue();
        void updateGradients(std::vector<double> expectedValue);
        
        double value, gradient, preActivationValue, bias, biasGradient;
    private:
        NeuralNetwork* network; //network it belongs to
        std::pair<int, int> location; //layer in network, location in layer
};