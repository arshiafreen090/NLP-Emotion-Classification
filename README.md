# NLP Emotion Classification

A machine learning project that predicts the emotion expressed in a piece of text.

## Overview

This project uses Natural Language Processing and Machine Learning to classify text into six emotion categories:

- 😔 Sadness
- 😠 Anger
- ❤️ Love
- 😮 Surprise
- 😨 Fear
- 😊 Joy

The final model uses TF-IDF for feature extraction and Logistic Regression for classification.

## Model

- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression
- 80/20 train-test split
- Test Accuracy: 86.16%

## Pipeline

```text
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Emotion Prediction
