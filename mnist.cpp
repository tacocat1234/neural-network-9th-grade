#include <fstream>
#include <stdexcept>
#include "mnist.hpp"

//idx files store their header ints as big endian
static int readBigEndianInt(std::ifstream& file){
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

trainingData loadMnistData(const std::string& imagePath, const std::string& labelPath, int maxSamples){
    std::ifstream images(imagePath, std::ios::binary);
    std::ifstream labels(labelPath, std::ios::binary);
    if (!images || !labels){
        throw std::runtime_error("could not open " + imagePath + " or " + labelPath);
    }

    //check magic numbers for idx3  = (images) and idx1 = (labels)
    if (readBigEndianInt(images) != 2051 || readBigEndianInt(labels) != 2049){
        throw std::runtime_error("bad MNIST file header");
    }
    int imageCount = readBigEndianInt(images);
    int rows = readBigEndianInt(images);
    int cols = readBigEndianInt(images);
    int labelCount = readBigEndianInt(labels);
    if (imageCount != labelCount){
        throw std::runtime_error("image and label counts do not match");
    }

    int count = (maxSamples >= 0 && maxSamples < imageCount) ? maxSamples : imageCount;
    int pixelCount = rows * cols;

    trainingData data;
    data.reserve(count);
    std::vector<unsigned char> pixels(pixelCount);
    for (int i = 0; i < count; i++){
        images.read(reinterpret_cast<char*>(pixels.data()), pixelCount);
        char label;
        labels.read(&label, 1);

        std::vector<double> input(pixelCount);
        for (int p = 0; p < pixelCount; p++){
            input[p] = pixels[p] / 255.0; //scale val 0-1
        }
        std::vector<double> expected(10, 0.0);
        expected[static_cast<int>(label)] = 1.0; //one hot
        data.push_back(std::make_pair(input, expected));
    }
    if (!images || !labels){
        throw std::runtime_error("MNIST files ended early");
    }
    return data;
}

int indexMax(const std::vector<double>& values){
    int best = 0;
    for (int i = 1; i < values.size(); i++){
        if (values[i] > values[best]){
            best = i;
        }
    }
    return best;
}
