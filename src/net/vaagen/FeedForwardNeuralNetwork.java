package net.vaagen;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by Magnus on 8/27/2018.
 */
public class FeedForwardNeuralNetwork {

    private float learningRate;
    private Random random;

    private Layer[] layers;
    private int[] layerSize;

    public FeedForwardNeuralNetwork(int[] layerSize, float learningRate, ActivationFunction activationFunction) {
        this.learningRate = learningRate;
        this.layerSize = layerSize;
        this.random = new Random();

        this.layers = new Layer[layerSize.length];
        populateLayers(activationFunction);
    }

    public void setActivationFunctionOutput(ActivationFunction activationFunctionOutput) {
        layers[layers.length-1].activationFunction = activationFunctionOutput;
    }

    private void populateLayers(ActivationFunction activationFunction) {
        for (int i = 0; i < layers.length; i++) {
            Layer prevLayer = (i > 0) ? layers[i-1] : null;
            Layer layer = new Layer(layerSize[i], prevLayer, activationFunction);
            layers[i] = layer;
        }
    }

    public float[] calc(float[] input) {
        // Fill the first layer with the input
        layers[0].fill(input);

        // Retrieve the output
        float[] output = layers[layers.length-1].calc();
        return output;
    }

    public void backPropagation(float[] expectedOutput) {
        // TODO : We can probably do the same with the last layer, call derive(), input float[] of just error and the activation function, which would be the derivedError (might be a problem at the last one tho)

        // Vi skal finne partiell deriverte av e / W1, som er det samme som: (e / OA1) * (OA1 / O1) * (O1 / W1), som vi løser hver for seg
        // Start from the error, work to the activated output, then the un-activated output, the the activated last hidden node, then the unactivated last hidden node, and so on

        // TODO : Reset the activatedValue? Or do that before calculation? Or after BackProp?

        // Manually derive the output layer
        // We want to set the weights before we can move on to the next step
        Layer outputLayer = layers[layers.length-1];
        for (int n = 0; n < outputLayer.nodes.length; n++) {
            Node node = outputLayer.nodes[n];

            // First derive the error
            float errorDerived = deriveError(expectedOutput[n], node.activatedValue, outputLayer.nodes.length);

            // Then derive the activation function
            float activationDerived = ActivationFunction.derive(outputLayer.activationFunction, node.value);
            node.derivedValue = errorDerived * activationDerived;

            // Loop through the last hidden layer to derive the weights
            for (int i = 0; i < outputLayer.prevLayer.nodes.length; i++) {
                // Stores the weight, and waits to update them till the end
                node.deriveForWeight(outputLayer.prevLayer, i, errorDerived, activationDerived);
            }

            // Derive for the bias as well
            outputLayer.deriveForBias(errorDerived, activationDerived, n);
        }

        // Continue on with the next layers
        Layer lastLayer = outputLayer;
        Layer currentLayer = outputLayer.prevLayer;
        while (currentLayer.prevLayer != null) {
            // Derive the layer
            currentLayer.derive(lastLayer);

            // Get the previous layer again
            currentLayer = currentLayer.prevLayer;
            lastLayer = lastLayer.prevLayer;
        }

        //System.out.println("Updating Weights!");

        // Update all the weights and bias weights
        // No reason to update the first layer as it holds neither bias nor weights
        for (int l = 1; l < layers.length; l++) {
            layers[l].updateWeightsAndBias(learningRate);
        }
    }

    public float error(float[] expectedOutput, float[] output) {
        if (expectedOutput.length != output.length)
            throw new Error("FeedForwardNeuralNetwork.error(): ExpectedOutput[] and Output[] are not same size!");
        // Mean Squared Error
        float error = 0;
        for (int i = 0; i < output.length; i++) {
            error += Math.pow(expectedOutput[i] - output[i], 2);
        }

        return error / output.length;
    }

    public float deriveError(float expectedOutput, float output, int layerLength) {
        float derivedError = - (2 / layerLength) * (expectedOutput - output);
        //System.out.println("Derived ERROR: " + derivedError + ", " + output);
        return derivedError;
    }

    public void print() {
        System.out.println("=============");
        System.out.println("Network Structure:");
        for (Layer layer : layers) {
            String s = "Layer: ";
            for (Node node : layer.nodes) {
                s += Arrays.toString(node.weights);
            }

            s += ", Bias: ";
            if (!layer.firstLayer) {
                s += Arrays.toString(layer.biasWeights);
            }


            System.out.println(s);
        }
    }

    public class Layer {

        private ActivationFunction activationFunction;
        private Layer prevLayer;
        private Node[] nodes;
        private boolean firstLayer;

        private float[] values;

        private float bias;
        private float[] biasWeights;
        private float[] newBiasWeights;

        public Layer(int length, Layer prevLayer, ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            this.firstLayer = prevLayer == null;
            this.prevLayer = prevLayer;

            // All except the first layer has a bias
            if (!firstLayer) {
                biasWeights = new float[length];
                newBiasWeights = new float[length];

                // Initialize weights with random value between 0 and 1
                for (int i = 0; i < biasWeights.length; i++) {
                    biasWeights[i] = random.nextFloat();
                }
            }
            bias = 1;

            nodes = new Node[length];
            populateNodes();
        }

        public void populateNodes() {
            for (int i = 0; i < nodes.length; i++) {
                Node node = new Node(prevLayer);
                nodes[i] = node;
            }
        }

        public float[] calc() {
            // values = [3, 5]
            // We don't want to apply any activation function, bias or weights BEFORE the first layer
            if (firstLayer) {
                // Update the values of the nodes!
                for (int n = 0; n < nodes.length; n++) {
                    nodes[n].value = values[n];
                    nodes[n].activatedValue = values[n];
                }
                return values;
            }

            // Get the values from the previous layer
            float[] prevLayerValues = prevLayer.calc();
            float[] output = new float[nodes.length];

            // Loop through the nodes of the previous layer and sum the weights
            for (int n = 0; n < nodes.length; n++) {
                for (int w = 0; w < nodes[n].weights.length; w++) {
                    output[n] += prevLayerValues[w] * nodes[n].weights[w];
                }
            }

            // Loop through the output nodes and add the bias, also call the activation function
            for (int i = 0; i < nodes.length; i++) {
                float value = output[i] + bias * biasWeights[i];
                float activatedOutput = ActivationFunction.activate(activationFunction, value);
                output[i] = activatedOutput;

                // Store the value and activated output so we can use it in backprop. later
                nodes[i].value = value;
                nodes[i].activatedValue = activatedOutput;
                //System.out.println("ActivatedValue: " + activatedOutput);
            }

            return output;
        }

        public void derive(Layer nextLayer) {
            //System.out.println("DERIVING!");
            for (int n = 0; n < nodes.length; n++) {
                Node node = nodes[n];

                // The total weight derived
                float derivedWeightSum = 0;

                // Loop through all the nodes of the previous layer to get their weights and divide by the activated value, which is the derived of the weight (I think)
                for (int i = 0; i < nextLayer.nodes.length; i++) {
                    // The derived value is where we stopped last loop
                    // Multiply this with the weight connecting the node in the next layer and this node
                    derivedWeightSum += nextLayer.nodes[i].derivedValue * nextLayer.nodes[i].weights[n];
                }

                // Then derive the activation function
                float activationDerived = ActivationFunction.derive(activationFunction, node.value);

                // Loop through the previous layer to derive the weights
                for (int i = 0; i < prevLayer.nodes.length; i++) {
                    // Stores the weight, and waits to update them till the end
                    node.deriveForWeight(prevLayer, i, derivedWeightSum, activationDerived);
                }

                // Store the derived value for the upcoming layer
                node.derivedValue = derivedWeightSum * activationDerived;

                // Derive for the bias as well
                deriveForBias(derivedWeightSum, activationDerived, n);
            }
        }

        public void deriveForBias(float derivedWeightSum, float derivedActivated, int node) {
            newBiasWeights[node] = derivedWeightSum * derivedActivated * bias;
        }

        public void updateWeightsAndBias(float learningRate) {
            // Update the bias
            for (int w = 0; w < biasWeights.length; w++) {
                biasWeights[w] -= learningRate * newBiasWeights[w];
                newBiasWeights[w] = 0;
            }

            // Update the weights
            for (int n = 0; n < nodes.length; n++) {
                nodes[n].updateWeights(learningRate);
            }
        }

        public void fill(float[] values) {
            this.values = values;
        }

    }

    public class Node {

        // NOTE: These weights go from the node, and to the left
        // This means that the input layer does not have any weights connected to them, they are connected from the first hidden layer instead
        private float[] weights;
        private float[] newWeights;
        private float value; // Value before activated
        private float activatedValue;
        private float derivedValue; // This is the derivedError * derivedActivation

        public Node(Layer prevLayer) {
            if (prevLayer != null) {
                weights = new float[prevLayer.nodes.length];
                newWeights = new float[prevLayer.nodes.length];

                // Initialize weights with random value between 0 and 1
                for (int i = 0; i < weights.length; i++) {
                    weights[i] = random.nextFloat();
                }
            }
        }

        public void deriveForWeight(Layer prevLayer, int node, float errorDerived, float activationDerived) {
            // INPUT LAYER HAS 0 AS ACTIVATED VALUE
            //System.out.println("Derive for weight: " + prevLayer.nodes[node].activatedValue + ", " + errorDerived + ", " + activationDerived);
            // Since this step is only to multiply the weight with the activated output, the derivative is the activated output.
            // Since there is only one node connected to every weight, we just store the weight and update it later
            newWeights[node] = prevLayer.nodes[node].activatedValue * errorDerived * activationDerived;
        }

        public void updateWeights(float learningRate) {
            //System.out.println("Old: " + Arrays.toString(weights) + " | New: " + Arrays.toString(newWeights));
            for (int w = 0; w < weights.length; w++) {
                weights[w] -= learningRate * newWeights[w];
                newWeights[w] = 0;
            }
        }

    }

}
