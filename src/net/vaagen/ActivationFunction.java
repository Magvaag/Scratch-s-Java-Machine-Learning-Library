package net.vaagen;

/**
 * Created by Magnus on 8/27/2018.
 */
public enum ActivationFunction {

    SIGMOID, LINEAR;

    public static float activate(ActivationFunction activationFunction, float value) {
        switch (activationFunction) {
            case SIGMOID:
                return activateSigmoid(value);
            case LINEAR:
                return activateLinear(value);
        }

        return activateSigmoid(value);
    }

    public static float derive(ActivationFunction activationFunction, float value) {
        switch (activationFunction) {
            case SIGMOID:
                return deriveSigmoid(value);
            case LINEAR:
                return deriveLinear(value);
        }

        return deriveSigmoid(value);
    }

    public static float activateSigmoid(float value) {
        return (float)(1 / (1 + Math.exp(-value)));
    }

    public static float activateLinear(float value) {
        return value;
    }

    public static float deriveSigmoid(float value) {
        // Not the problem
        // Paper shows: (float)((1 / (1 + Math.exp(-value))) * (1 - 1 / (1 + Math.exp(-value))));
        // The correct derivative: (float) (Math.exp(-value) / Math.pow(1 + Math.exp(-value), 2));
        float derivative = (float)(Math.exp(-value) / Math.pow(1 + Math.exp(-value), 2));//(float) (Math.exp(-value) / Math.pow(1 + Math.exp(-value), 2));
        //System.out.println("DeriveSigmoid: " + value + " = " + deriviative);
        return derivative;
    }

    public static float deriveLinear(float value) {
        return 1;
    }

}
