#include "NerualNetwork.h"
#include <ctime>

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

}