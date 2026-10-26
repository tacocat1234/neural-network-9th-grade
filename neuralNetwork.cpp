#include <random>
#include "neuralNetwork.hpp"
#include "functions.hpp"

NeuralNetwork::NeuralNetwork(std::vector<double> layerSizes, functionPtr activation, functionPtr activationDerivative, double learningRate, int batchSize) : layerSizes(layerSizes), activationFunction(activation), activationDerivative(activationDerivative), learningRate(learningRate), batchSize(batchSize){
    // Create a random number engine rd
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister pseudo-random number generator
    const double mean = 0.0; //consts
    double stddev; // 2 divided by number of input nodes + number of output nodes
    
    
    nodes.resize(layerSizes.size());
    weights.resize(layerSizes.size() - 1);
    
    for (int l = 0; l < layerSizes.size(); l++){ //iterate through each element of layer sizes (aka each layer)
        stddev = std::sqrt(2.0/((l > 0 ? layerSizes[l - 1] : 0) + (l < layerSizes.size() - 1 ? layerSizes[l + 1] : 0)));
        std::normal_distribution<double> dist(mean, stddev);
        
        for(int n = 0; n < layerSizes[l]; n++){ //iterates through nodes  in the layer
            nodes[l].push_back(
                new Node(this, std::make_pair(l, n), //new node who belongs to *this*, in layer l (which iterates through all layers) and in location i (which iterates through number of nodes that should be in each layer) along the layer
                dist(gen), dist(gen))); //whose value and bias are randomly init based on the previously defined distributiuon
            if (l < layerSizes.size() - 1){ //if this layer isnt the output layer
                weights[l].resize(layerSizes[l]); //resize each item repersenting a layer to the size of that layer.
                for(int w = 0; w < layerSizes[l + 1]; w++){ //iterates through the nodes of the next layer
                    weights[l][n].push_back(dist(gen));
                    //the weight coming from the node in layer "l" at position "n" (see previous) that goes to node "w" of the next layer is initialized to a random value based on normal distribution 
                    //[which layer is in][which node from that layer][which node that the weight is going to from that node]
                }
            }
        }
    }
    nodeValues.resize(layerSizes.size());
    nodeGradients.resize(layerSizes.size()); //the 1st degree of the node gradients is equal to the number of layers. at this point, nodegradients is a vector with a number of items equal to the numbers of layers in the network
    biasGradients.resize(layerSizes.size()); //see above
    weightGradients.resize(layerSizes.size()); //weightGradients repersents a vector of a bunch of layers
    for (int i = 0; i < layerSizes.size(); i++){
        nodeValues[i].resize(layerSizes[i]);
        nodeGradients[i].resize(layerSizes[i], 0);//the size of the vector repersenting the layer is equal to the size of the layer as defined by layerSizes.size()
        biasGradients[i].resize(layerSizes[i], 0);
        if (i != layerSizes.size() - 1){ // if not output layer
            weightGradients[i].resize(layerSizes[i]);//resize the layersizes in wGrad to be acutal layer sizes
            //weightGradients[i] represents a vector of a bunch of nodes 
            for (auto& weightsFromNode : weightGradients[i]){//iterate through each item in weightGradients[i] (each node in the layer)
                weightsFromNode.resize(layerSizes[i + 1], 0); //make the # of weights from that node be the number of nodes that recieve an input from that node; each of those items reperesnts a vector of all weights leavign from the current node
            }
        }
    }
    
}

NeuralNetwork::~NeuralNetwork(){ //destructor
    for (std::vector<Node*>& layer : nodes){ //for each layer of nodes
        for (Node* nodePtr : layer){ //for each node in said layer
            delete nodePtr; //we delete that node
        }
    }
}

void NeuralNetwork::passInputs(std::vector<double> inputs){
    inputs.resize(layerSizes[0]);
    int i = 0;
    for (double iValue : inputs){
        nodes[0][i]->value = iValue;
        i++;
    }
}

void NeuralNetwork::updateAll(){
    for (int l = 1; l < layerSizes.size(); l++){ //for each non input layer
        for (int n = 0; n < layerSizes[l]; n++){ //for each node in said layers
            nodes[l][n]->updateValue();
        }
    }    
}

std::vector<double> NeuralNetwork::outputValues(){
    std::vector<double> output;
    for (auto* node : nodes[layerSizes.size() - 1]){
        output.push_back(node->value);
    }
    return output;
}

std::vector<double> NeuralNetwork::evaluate(std::vector<double> inputs){
    passInputs(inputs);
    updateAll();
    return outputValues();
}

void NeuralNetwork::updateWeightsAndBiases() {//SGD updating
    for (int l = 0; l < layerSizes.size() - 1; l++) {
        for (int n = 0; n < layerSizes[l]; n++) {
            for (int nextN = 0; nextN < layerSizes[l + 1]; nextN++) {
                // Update weights
                weights[l][n][nextN] -= learningRate * weightGradients[l][n][nextN];
            }
            // Update biases
            nodes[l][n]->bias -= learningRate * biasGradients[l][n];
        }
    }
}

void NeuralNetwork::backpropagate(trainingData& dataSample){
    //reset grads from prev batches
    std::vector<std::vector<double>> nodeGradientSums;
    nodeGradientSums.reserve(nodeGradients.size()); //one vec per layer
    for (auto& layer : nodeGradients){
        nodeGradientSums.emplace_back(layer.size(), 0.0);
    }
    std::vector<std::vector<double>> biasGradientSums;
    biasGradientSums.reserve(biasGradients.size());
    for (auto& layer : biasGradients){
        biasGradientSums.emplace_back(layer.size(), 0.0);
    }
    networkWeights weightGradientSums;
    weightGradientSums.reserve(weightGradients.size());
    for (auto& layer : weightGradients){
        weightGradientSums.emplace_back(); 
        weightGradientSums.back().reserve(layer.size());
        for (auto& weightsFromNode : layer){
            weightGradientSums.back().emplace_back(weightsFromNode.size(), 0.0);
        }
    }

    for (auto& dataPoint : dataSample){
        passInputs(dataPoint.first);
        updateAll();
        std::vector<double> expectedValues = dataPoint.second;
        
        for (int l = layerSizes.size() - 1; l >= 0; l--){//for each layer, backwards
            for(int n = 0; n < layerSizes[l]; n++){//for each node in layer
                //input nodes have no bias and nothing uses their gradients. Skipping them
                // saves a lot of training time.
                if (l > 0){
                    nodes[l][n]->updateGradients(expectedValues); //node updateGradients, updates personal and global of network
                    nodeGradientSums[l][n] += nodeGradients[l][n];
                    biasGradientSums[l][n] += biasGradients[l][n];
                }
                //update weightGradients next
                if (l < layerSizes.size() - 1){//if not last layer
                    for (int nextN = 0; nextN < layerSizes[l + 1]; nextN++){//for each node in the next
                        // Node::updateGradients() already computes
                        // biasGradient = network->activationDerivative(preActivationValue) * gradient;
                        weightGradients[l][n][nextN] = nodes[l][n]->value * //deriv. of next node's preActVal w/respect to this weight
                                nodes[l + 1][nextN]->biasGradient; //activation deriv. of next node times grad of next node w/respect to output
                        weightGradientSums[l][n][nextN] += weightGradients[l][n][nextN];
                    }
                }
            }
        }
    }
    for (int l = 0; l < nodeGradientSums.size(); l++){
        for (int n = 0; n < nodeGradientSums[l].size(); n++){
            biasGradients[l][n] = biasGradientSums[l][n]/dataSample.size();
            if(l + 1 < nodeGradientSums.size()){
                for (int nextN = 0; nextN < nodeGradientSums[l + 1].size(); nextN++){
                    weightGradients[l][n][nextN] = weightGradientSums[l][n][nextN]/dataSample.size();
                }
            }
        }
    }
    updateWeightsAndBiases();
}

void NeuralNetwork::train(int iterations, trainingData& dataSet){
    std::vector<trainingData> miniBatches = formMiniBatches(batchSize, dataSet);
    
    int i = 0;
    while (i < iterations){
        for (auto& miniBatch : miniBatches){
            if (i >= iterations){break;}
            backpropagate(miniBatch);
            i++;
        }
    }
}