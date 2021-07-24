#ifndef NerualNetworkH
#define NerualNetworkH

#include <vector>
#include <string>

class NerualNetwork {

    private:
        std::vector<double> inputs;
        std::vector<std::vector<double>> hiddenLayer;
        std::vector<double> outputs;
        std::vector<std::vector<std::vector<double>>> weights;

        void init(int numOfInputs, int numOfHiddenLayers, std::vector<int> nodesInHiddenLayer, int numOfOutputs);

    public:
        NerualNetwork(int numOfInputs, int numOfHiddenLayers, std::vector<int> nodesInHiddenLayer, int numOfOutputs);
        void setInputs(std::vector<double> inputs);
        int think();
        void saveWeights(std::string filepath);

};

#endif