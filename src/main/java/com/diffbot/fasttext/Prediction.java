package com.diffbot.fasttext;

public class Prediction {
    float probability;
    String label;

    public Prediction(float probability, String label) {
        this.probability = probability;
        this.label = label;
    }

    public float getProbability() {
        return probability;
    }

    public String getLabel() {
        return label;
    }
}
