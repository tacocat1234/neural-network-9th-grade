#pragma once

#include <string>
#include <vector>
#include "using.hpp"

//inputs are the 784 pixels scaled to 0-1, expected outputs are a one hot vector size 10
//maxSamples < 0 to load all
trainingData loadMnistData(const std::string& imagePath, const std::string& labelPath, int maxSamples = -1);

int indexMax(const std::vector<double>& values);
