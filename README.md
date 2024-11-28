# Wav2Vec Conformer MoE Model Fine-Tuning and Optimization

## Overview

This Jupyter notebook demonstrates the process of fine-tuning and optimizing the Wav2Vec2 model for speech-to-text tasks using the Common Voice dataset. The notebook also includes a comparison of the performance of the Wav2Vec2 model with the Whisper-Turbo model, specifically focusing on the Word Error Rate (WER) metric.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Pre-Processing](#pre-processing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Comparison of WER](#comparison-of-wer)
8. [Conclusion](#conclusion)

## Introduction

The goal of this notebook is to fine-tune the Wav2Vec2 model on the Urdu subset of the Common Voice dataset and evaluate its performance. Additionally, we will compare the WER of the Wav2Vec2 model with that of the Whisper-Turbo model to assess their relative effectiveness in transcribing audio.

## Requirements

To run this notebook, you need the following Python packages:

- `torch`
- `torchaudio`
- `transformers`
- `datasets`
- `sentencepiece`
- `jiwer`
- `librosa`
- `requests`
- `numpy`
- `scipy`

You can install these packages using the following command:
write me a full detailed readme file for this notebook and also tell about the comparison of wer between both models

- bash
- !pip install torch torchaudio transformers datasets sentencepiece jiwer librosa requests numpy scipy


## Dataset

The dataset used in this notebook is the Common Voice dataset provided by Mozilla. We specifically use the Urdu subset for training and evaluation. The dataset is loaded using the `datasets` library.

## Pre-Processing

The pre-processing steps include:

1. Loading the dataset.
2. Limiting the dataset to a specified percentage for training, validation, and testing.
3. Preprocessing the audio data to extract input values and labels.
4. Removing unnecessary columns from the dataset.

## Model Training

The Wav2Vec2 model is fine-tuned using the following steps:

1. Loading the pre-trained Wav2Vec2 model and processor.
2. Setting up training arguments, including batch size, number of epochs, and logging strategies.
3. Training the model using the `Trainer` class from the `transformers` library.

## Model Evaluation

After training, the model is evaluated on the test dataset. The evaluation includes generating predictions for the test audio samples and calculating the WER.

## Comparison of WER

The WER for both models is calculated as follows:

- **Wav2Vec2 Model WER**: The WER for the Wav2Vec2 model is approximately **52.63%**.
- **Whisper-Turbo Model WER**: The WER for the Whisper-Turbo model is significantly lower at approximately **8.90%**.

This indicates that the Whisper-Turbo model performs better in terms of transcription accuracy compared to the Wav2Vec2 model on the same dataset.

## Conclusion

In this notebook, I successfully fine-tuned the Wav2Vec2 model for speech-to-text tasks on Urdu Data and compared its performance with the Whisper-Turbo model. The results show that while Wav2Vec2 is a powerful model, the Whisper-Turbo model provides superior transcription accuracy as indicated by the lower WER.

For further improvements, consider experimenting with different hyperparameters, larger datasets, or additional pre-processing techniques to enhance the performance of the Wav2Vec2 model.
