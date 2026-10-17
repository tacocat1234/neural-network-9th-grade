#include "nodes.hpp"
#include "functions.hpp"

Node::Node(NeuralNetwork* network, std::pair<int, int> location, double value, double bias) : network(network), location(location), value(value), bias(bias){
}//just pass all values 

void Node::updateValue(){ //function that updates value of node inPlace, based on previous layer and weights
    if (location.first != 0){ //if not input
        value = 0; //init to 0 for summation
        for (int i = 0; i <
            network->nodes[location.first - 1].size(); 
            i++){ //for each in previous layer
            //value += prevNode[i]*weightFromPrevNode;
            value += network->nodes[location.first - 1][i]->value * // the node at location i in the layer before this 
                network->weights[location.first - 1][i][location.second]; //multiplied by the weight starting from the node at location i in the layer that is 1 before this that goes to this node
        }
        value += bias;//add bias in
        preActivationValue = value;
        value = activationFunction(value); //evaluated at the activation function 
        network->nodeValues[location.first][location.second] = value;
    }
}

void Node::updateGradients(std::vector<double> expectedValues){
    if (location.first == network->layerSizes.size() - 1){//if last layer
        std::vector<double> outputs = network->outputValues();
        
        gradient = meanSquareDerivative(outputs, expectedValues, location.second); //set self value
        
        network->nodeGradients[location.first][location.second] = gradient; //set network value
        
            //below is db/dn of bias with respect to node value
        biasGradient = activationDerivative(preActivationValue) * 
            gradient; //dn/do derivative of this node with respect to output
            
        network->biasGradients[location.first][location.second] = biasGradient; //set network value
    } else{
        double partialDerivative = 0;
        for (int nextN = 0; nextN < network->layerSizes[location.first + 1]; nextN++){
            partialDerivative += 
                activationDerivative(
                    network->nodes[location.first + 1][nextN]->preActivationValue) * 
                //da/ds derivative of activation with resepect to sum times \/ derivative of sum
                network->weights[location.first][location.second][nextN] * //weight coming from this position to the nextN location in the next layer
                    //first bit is just the result after chain rule for finding derivative of this node with respect to some node in the next layer at nextN.
                network->nodeGradients[location.first + 1][nextN]; //multiplied by partial derivative of that node at nextN with respect to the output (chain rule)
        }
        gradient = partialDerivative;
        network->nodeGradients[location.first][location.second] = gradient;
        
        biasGradient = activationDerivative(preActivationValue) * gradient;
        network->biasGradients[location.first][location.second] = biasGradient;
    }
}