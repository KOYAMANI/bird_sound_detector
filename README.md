## Overview

This project aims to build a machine learning model to detect bird species by their voice.

Input: Mel spectrogram image dimensions 48 x 128

Output: True or False, described in dimensions 1 x 95 (total numbers of species)

Evaluation metric: We chose accuracy and F1 score since the project focus is the species detection so that it is not too critical problem to have FP or FN.

## Dataset

We use the dataset from [Kaggle](https://www.kaggle.com/competitions/birdclef-2021/data). This dataset is a standardized version of some of the bird call audio data in [XenoCanto](https://xeno-canto.org/)

### train_short_audio

The bulk of the training data consists of short recordings of individual bird calls generously uploaded by users of [XenoCanto](https://xeno-canto.org/).org. These files have been downsampled to 32 kHz where applicable to match the test set audio and converted to the ogg format. The training data should have nearly all relevant files; we expect there is no benefit to looking for more on XenoCanto

### train_metadata.csv

A wide range of metadata for the training data.

The most directly relevant fields are:

- `primary_label`: a code for the bird species. You can review detailed information about the bird codes by appending the code to https://ebird.org/species/, such as https://ebird.org/species/amecro for the American Crow.
- `author`: the user who provided the recording.
- `latitude & longitude`: coordinates for where the recording was taken. Some bird species may have local call ‘dialects,’ so you may want to seek geographic diversity in your training data.
- `date`: while some bird calls can be made year round, such as an alarm call, some are restricted to a specific season. You may want to seek temporal diversity in your training data.
- `filename`: the name of the associated audio file.
- `scientific_name & common_name`: Scientific terms and common terms of bird name
- `time `: Time in the day when the audio was recorded
- `url` : original recording data url specified in xeno-canto
- `rating` : the quality of the audio recording

### train_soundscape_labels.csv

- `row_id`: ID code for the row.

- `site` Site ID.

- `seconds`: the second ending the time window

- `audio_id`: ID code for the audio file.

- `birds`: space delimited list of any bird songs present in the 5 second window. The label nocall means that no call occurred.

### Data Exploration

We performed data exploration in the [data_exploratory.ipynb](https://github.com/KOYAMANI/bird_sound_detector/blob/main/data_exploratory.ipynb).

In the result of the exploration, we have discovered that there are a total 397 species with in the dataset. And the distribution is as follows; 28 species with less than 50 recordings, 300 species with between 50 and 200 recordings, and 20 species with over 500 recordings.

In addition, the dataset ranging in size from 6kb to 32mb of recording time.
More than 95% of files are smaller than 5000kb

These result illustrates that the dataset is not balanced so that the number of data needs to be standardized before moving on to the machine learning.

Therefore we decided to narrow them down to 95 species/11692 audio recordings based on species popularity and audio quality to proceed further processes.

## Candidates

### MLP (Multi-layer perceptron)

In the early work, applying NNs to animal sound used the basic MLP archetecture. It uses manually-designed summary features (such as syllable duration, peak frequency) as an input. However it is superseded and dramatically outperformed by CNN and (to a lesser extent) recurrent neural network (RNN) architectures. Because they can take advantage of the sequential/grid structure in raw or lightly-preprocessed data, so the input to CNN/RNN can be time series or time-frequency spectrogram data.

### CNN (Convolutional Neural Network)

In CNN, acoustic data is reduced to a small number of summary features in a manually-designed feature extraction process—keeps the input in a much higher dimensional format, allowing for much richer information to be presented. CNN having many fewer free parameters than the equivalent MLP, thus being easier to train.

### RNN (Recurrent Neural Network)

It is widely understood that sound scenes and vocalisations can be driven by dependencies over both short and very long timescales. This consideration about time series in general was the inspiration for the design of recurrent neural networks.It has the capacity to pass information forwards (and/or backwards) arbitrarily far in time while making inferences. Hence, RNNs have often been explored to process sound, including animal sound.

### CRNN

In around 2017 it was observed that adding an RNN layer after the convolutional layers of a CNN could give strong performance in multiple audio tasks, with an interpretation that the RNN layer(s) perform temporal integration of the information that has been preprocessed by the early layers. However, CRNNs can be more computationally intensive to train than CNNs, and the added benefit is not universally clear.

## Solution

We chose CNN for these reasons:

- Our input data is mel spectrogram of audio, and CNN is usually utilized for image recognition, pattern recognition and computer vision.
- CNN is considered as more powerful tool than RNN. RNN has less features and low capabilities compeared to CNN.
- CNNs are now dominant. There are a lot off-the-shelf CNN architectures can be used and evaluated.

### Hyperparameters

- `activation function`: ReLU
- `optimizer`: Adam
- `number of epochs`: 25
- `learning rate`: 0.001
- `dropout`: 0.5
- `batch size`: 32
- `number of layers and units` : 4 convolutional layers + 2 fully-connected layers. Each input data has a shape of (48,128), and is a 5 second mel-spectrogram.

## Evaluation

### Benchmarks

We evaluated the performance of our CNN model with 4 other benchmarks as following

- variant 2 : Changed activation function from relu to LeakyReLU

- variant 3: changed kernel_size from (3,3) to (3,5)

- variant 4: Changed activation function from relu to tanh

- Random: Guess species by random guess

### Evaluation metrics

We decided to focus accuracy and F1 score. Because in the typical sound classification project, precision and recall are equally important.

- accuracy: measures the number of predictions that are correct as a percentage of the total number of predictions that are made.
- F1 score: the harmonic mean of precision and recall. F1 score has been designed to work well on imbalanced data.

## Result

We investigated all the variants above (n=3, max epochs=25) and the result of evaluation metrics are as follows.

Based on the mean value, v2 showed the best results in both accuracy and F1 score.
v1, 3, and 4 were nearly the same in terms of accuracy. In terms of F1 score, however, v3 came in second with 0.055, followed by v1 with 0.038, and v4 recorded only 0.01.

Random guess always scored nearly 0.01 as it should behave

![result_1](https://github.com/KOYAMANI/bird_sound_detector/blob/main/images/result_1.png)

We also increased the maximum number of epochs from 25 to 50 and tried all variations. This was because we expected especially v3 to require more training time, since we changed the karnel size from (3,3) to (3,5).

In this condition, v2 still scored the best in both accuracy and f1 score.
In terms of accuracy, v3 has improved by almost 50% compared to the previous condition, while v1 and v4 have only improved by 25%.
About the F1 score, v3 recoded 0.13, followed by v1 with 0.080, and v4 resulted 0.031.

![result_2](https://github.com/KOYAMANI/bird_sound_detector/blob/main/images/result_2.png)

Overall, while v3 showed a significant performance improvement with the increase in epochs, v2 exceeded it and always performed the best.
