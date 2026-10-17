#include <vector>
#include "using.hpp"

//double activationFunction(double stimulus);
double tanhDerivative(double stimulus);

double sigmoid(double stimulus);
double sigmoidDerivative(double stimulus);

double relu(double stimulus);
double reluDerivative(double stimulus);

std::vector<trainingData> formMiniBatches(int batchSize, trainingData& dataSet);

double meanSquareError(std::vector<double> output, std::vector<double> expected);
double meanSquareDerivative(std::vector<double> output, std::vector<double> expected, int indexOf);
