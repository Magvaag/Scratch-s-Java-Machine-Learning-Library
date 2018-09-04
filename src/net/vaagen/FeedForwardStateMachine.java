package net.vaagen;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Magnus on 8/31/2018.
 */
public class FeedForwardStateMachine {

    public List<float[]> inputArray;
    public List<Integer> decisionArray;

    public FeedForwardStateMachine() {
        reset();
    }

    public void saveState(float[] input, int output) {
        inputArray.add(input);
        decisionArray.add(output);
    }

    public void reset() {
        inputArray = new ArrayList<>();
        decisionArray = new ArrayList<>();
    }

}
