import net.vaagen.ActivationFunction;
import net.vaagen.FeedForwardNeuralNetwork;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by Magnus on 8/27/2018.
 */
public class XOR {

    // https://dzone.com/articles/the-very-basic-introduction-to-feed-forward-neural

    public static void main(String[] args) {
        new XOR();
    }

    public XOR() {
        // XOR a XOR network
        int[] networkStructure = new int[] {2, 2, 1};
        float learningRate = .1f;
        Random random = new Random();

        FeedForwardNeuralNetwork ffnn = new FeedForwardNeuralNetwork(networkStructure, learningRate, ActivationFunction.SIGMOID);

        // Using the linear activation function will let you come closer to the values 0 and 1
        // For that reason the weights won't explode as well to reach those numbers
        ffnn.setActivationFunctionOutput(ActivationFunction.LINEAR);

        ffnn.print();

        int printFrequency = 100;
        int episodes = 1000;

        for (int i = 0; i < 1000000; i++) {
            // Set the input values
            float[] input = {random.nextInt(2), random.nextInt(2)}; // random.nextInt(2), random.nextInt(2)
            float[] expectedOutput = { XOR(input[0], input[1]) };

            // Calculate the output
            float[] output = ffnn.calc(input);
            float error = ffnn.error(expectedOutput, output);


            ffnn.backPropagation(expectedOutput);

            if (i % printFrequency == 0){
                printError(i, input, output, expectedOutput, error);
                ffnn.print();
            }

            if (Float.NaN == output[0]) {
                System.out.println("BREAK!");
                break;
            }
        }

        // XOR the network to end the training
        testXOR(ffnn);
    }

    private static void testXOR(FeedForwardNeuralNetwork feedForwardNeuralNetwork) {
        System.out.println("===============");
        System.out.println("Testing network!");
        System.out.println("1, 0: " + Arrays.toString(feedForwardNeuralNetwork.calc((new float[] {1.0f, 0.0f}))));
        System.out.println("1, 1: " + Arrays.toString(feedForwardNeuralNetwork.calc((new float[] {1.0f, 1.0f}))));
        System.out.println("0, 1: " + Arrays.toString(feedForwardNeuralNetwork.calc((new float[] {0.0f, 1.0f}))));
        System.out.println("0, 0: " + Arrays.toString(feedForwardNeuralNetwork.calc((new float[] {0.0f, 0.0f}))));
    }

    private static void printError(int i, float[] input, float[] output, float[] expectedOutput, float error) {
        System.out.println("================ " + i);
        System.out.println("Input: " + Arrays.toString(input));
        System.out.println("Output: " + Arrays.toString(output));
        System.out.println("Expected Output: " + Arrays.toString(expectedOutput));
        System.out.println("Error: " + error);
    }

    private float XOR(float a, float b) {
        // For the sake of simplicity we don't check for a, b != 0, 1
        return a != b ? 1 : 0;
    }

}
