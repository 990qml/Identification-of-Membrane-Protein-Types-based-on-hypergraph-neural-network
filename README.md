# Identification of Membrane Protein Types based on hypergraph neural network

## Introduction
 In this study, Average block (AvBlock), discrete cosine transform (DCT), discrete wavelet transform (DWT), histogram of orientation gradients (HOG) and pseudo-PSSM (PsePSSM) are used to extract evolutionary features, next, we propose a hypergraph neural network model (HGNN) for integrating five features to identify membrane protein types. For performance evaluation, the proposed method was tested on four membrane protein benchmark Datasets. 
 In this repository, we release the code for train and test HGNN on four dataset.
 ## Installation
1、Environment configuration requirements: Install (Pytorch 0.4.0) (https://pytorch.org/). Also need to install yaml.The code has been tested with Python 3.8.

2、Datasets: Processed datasets are in the data folder and the membrane protein feature representation algorithm is in the feature folder. Data processing and configuration are described in the datasets folder

3、Run: Configuration file:config.yaml,D4 is the target dataset and can be replaced train.py: select the data to be trained, and then start training. See the code comments for the specific process, save the prediction results to A.csv of the corresponding dataset in data/protein
