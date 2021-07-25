#include "NerualNetwork.h"
#include <ctime>
#include <cmath>

NerualNetwork::NerualNetwork(int numOfInputs, int numOfHiddenLayers, std::vector<int> nodesInHiddenLayer, int numOfOutputs) {

    init(numOfInputs, numOfHiddenLayers, nodesInHiddenLayer, numOfOutputs);

}

void NerualNetwork::init(int numOfInputs, int numOfHiddenLayers, std::vector<int> nodesInHiddenLayer, int numOfOutputs) {

    srand(time(NULL));

    // init inputs
    for (int i = 0; i < numOfInputs; i++) {

        inputs.push_back(0);

    }

    // init hidden layer
    for (int i = 0; i < numOfHiddenLayers; i++) {

        std::vector<double> hiddenLayerSet;
        for (int k = 0; k < nodesInHiddenLayer[i]; k++) {

            hiddenLayerSet.push_back(0);

        }

        hiddenLayer.push_back(hiddenLayerSet);

    }

    // init outputs
    for (int i = 0; i < numOfOutputs; i++) {

        outputs.push_back(0);

    }

    // init weights
    for (int i = 0; i < hiddenLayer.size(); i++) {

        std::vector<std::vector<double>> weightSetBetweenLayer;

        // if it is the first set of connections and more than one hidden layer : create the weights
        if (i == 0) {

            for (int k = 0; k < inputs.size(); k++) {

                std::vector<double> weightsForInput;
                for (int l = 0; l < hiddenLayer[i].size(); l++) {

                    double randomWeight = ((double) rand() / RAND_MAX) * 2 - 1;
                    weightsForInput.push_back(randomWeight);

                }

                weightSetBetweenLayer.push_back(weightsForInput);

            }

        // if it is between two hidden layers : create the weights
        } else {

            for (int k = 0; k < hiddenLayer[i - 1].size(); k++) {

                std::vector<double> weightsForHiddenLayer;
                for (int l = 0; l < hiddenLayer[i].size(); l++) {

                    double randomWeight = ((double) rand() / RAND_MAX) * 2 - 1;
                    weightsForHiddenLayer.push_back(randomWeight);

                }

                weightSetBetweenLayer.push_back(weightsForHiddenLayer);

            }

        }

        weights.push_back(weightSetBetweenLayer);

    }

    std::vector<std::vector<double>> weightSetBetweenLayer;

    // create weights if there was no hidden layers
    if (hiddenLayer.size() == 0) {

        for (int i = 0; i < inputs.size(); i++) {

            std::vector<double> weightsForInputs;
            for (int k = 0; k < outputs.size(); k++) {

                double randomWeight = ((double) rand() / RAND_MAX) * 2 - 1;
                weightsForInputs.push_back(randomWeight);

            }

            weightSetBetweenLayer.push_back(weightsForInputs);

        }

    // final connection set if there was a hidden layer
    } else {

        for (int i = 0; i < hiddenLayer[hiddenLayer.size() - 1].size(); i++) {

            std::vector<double> weightsForHiddenLayer;
            for (int k = 0; k < outputs.size(); k++) {

                double randomWeight = ((double) rand() / RAND_MAX) * 2 - 1;
                weightsForHiddenLayer.push_back(randomWeight);

            }

            weightSetBetweenLayer.push_back(weightsForHiddenLayer);

        }

    }

    weights.push_back(weightSetBetweenLayer);

    // set defualt activation function
    activationFunctionType = "sigmoid";

}

double NerualNetwork::activationFunction(double x) {

    // return sigmoid value
    if (activationFunctionType == "sigmoid") {

        return 1 / (1 + std::exp(-1 * x)); 

    } else if (activationFunctionType == "tanh") {

        return std::tanh(x);

    }

    return 0;

}

std::vector<double> NerualNetwork::think() {

    // loop for every hidden layer
    for (int i = 0; i < hiddenLayer.size(); i++) {

        // do the math with inputs instead of a hidden layer if first set of connections
        if (i == 0) {

            for (int k = 0; k < hiddenLayer[i].size(); k++) {

                double total = 0.0;
                for (int l = 0; l < inputs.size(); l++) {

                    total += inputs[l] * weights[i][l][k];

                }

                hiddenLayer[i][k] = activationFunction(total);

            }

        
        // if it is between to hidden layers
        } else {

            for (int k = 0; k < hiddenLayer[i].size(); k++) {

                double total = 0.0;
                for (int l = 0; l < hiddenLayer[i - 1].size(); l++) {

                    total += hiddenLayer[i - 1][l] * weights[i][l][k];

                }

                hiddenLayer[i][k] = activationFunction(total);

            }

        }

    }

    // if there was no hidden layer
    if (hiddenLayer.size() == 0) {

        for (int i = 0; i < outputs.size(); i++) {

            double total = 0.0;
            for (int k = 0; k < inputs.size(); k++) {

                total += inputs[k] * weights[0][k][i];

            }

            outputs[i] = activationFunction(total);

        }

    // final connection between hidden layer and outputs
    } else {

        for (int i = 0; i < outputs.size(); i++) {

            double total = 0.0;
            for (int k = 0; k < hiddenLayer[hiddenLayer.size() - 1].size(); k++) {

                total += hiddenLayer[hiddenLayer.size() - 1][k] * weights[weights.size() - 1][k][i];

            }

            outputs[i] = activationFunction(total);

        }

    }

    return outputs;

}