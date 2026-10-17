using weightsFromNode = std::vector<double>;
using weightsFromLayer = std::vector<weightsFromNode>;
using networkWeights = std::vector<weightsFromLayer>;

using trainingData = std::vector<std::pair<std::vector<double>, std::vector<double>>>;

using functionPtr = double(*)(double);