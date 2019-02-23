[//]: # (Image References)

[image1]: ./asr_capstone_project/images/pipeline.png "ASR Pipeline"
[image2]: ./asr_capstone_project/images/loss.png "Loss"

# Natural Language Processing Nanodegree Project

## Inroduction

Collection of python NLP projects created for Udacity NLP Nanodegree course. 
It have practical projects in the field of NLP using modern libraries and frameworks such as Keras and Tensorflow.

## Setup

### Install
- Pomegranate 0.9.0
- Python 3
- NumPy
- TensorFlow 1.x
- Keras 2.x

### Start

All projects are within a [Jupyter Notebook](http://jupyter.org/). To start the notebook, run the command `jupyter notebook project_name.ipynb` in the corresponding folder.

## Projects

### Part of Speech Tagger 

This notebook uses the [Pomegranate](https://github.com/jmschrei/pomegranate) library to build a hidden Markov model for part of speech tagging with a [universal tagset](http://www.petrovi.de/data/universal.pdf).

#### Models
1. Most Frequent Class Tagger: training accuracy: 95.72%; testing accuracy: 93.01%
2. Hidden Markov Model Tagger: training accuracy: 97.54% testing accuracy: 95.95%

### Machine Translation

This notebook has different type of machine translation models starting from simple models to advanced models with good accuracy.
The model trained on small data to translate from English language to French, it can be expanded to train on more data or translate other languages.

#### Models
Training accuracy:
1. Simple Model: 62.12% 
2. Embedding Model: 76.86% 
3. Bidirectional Model: 66.50% 
4. Encoder-Decoder Model: 61.68% 
5. **Final Model** (Multiple Techniques): 96.74% 

### Automatic Speech Recognition 

Capstone project of the course with different NLP models and advanced techniques. This notebook builds a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline.

![ASR Pipeline][image1]

*'Reviewer feedback on Capstone Project: Impressive work in this resubmission and overall! I especially enjoyed seeing your predicted transcription which your Notebook printed in the last step "3: Obtain predictions"!'*

#### Models

Model 0: RNN loss: 779.6107 - val_loss: 759.2642

Model 1: RNN + TimeDistributed Dense loss: 143.2519 - val_loss: 150.7867

Model 2: CNN + RNN + TimeDistributed Dense loss: 107.3945 - val_loss: 145.1359

Model 3: Deeper RNN + TimeDistributed Dense loss: 135.6761 - val_loss: 146.6482

Model 4: Bidirectional RNN + TimeDistributed Dense loss: 159.4280 - val_loss: 162.5575

**Final Model**: CNN + MultiHeadSelfAttention + RNN + TimeDistributed Dense loss: 94.9789 - val_loss: 137.3039

![Loss][image2]

## Contribution
TODO and Implement parts in the notebook done by Zhaoning Li. Document strcure done by Udacity Team.
