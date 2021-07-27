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
        std::string activationFunctionType;

        void init(int numOfInputs, int numOfHiddenLayers, std::vector<int> nodesInHiddenLayer, int numOfOutputs);
        double activationFunction(double x);

    public:
        NerualNetwork(int numOfInputs, int numOfHiddenLayers, std::vector<int> nodesInHiddenLayer, int numOfOutputs);
        void setInputs(std::vector<double> in) {inputs = in;};
        void setActivationFunctionType(std::string func) {activationFunctionType = func;}
        std::vector<double> think();
        void saveData(std::string filepath);

};

#endif