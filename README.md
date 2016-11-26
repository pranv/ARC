# ARC
Code repository for reproducing the results in the paper - [Attentive Recurrent Comparators](http://openreview.net/forum?id=BJjn-Yixl "Paper on OpenReview")

## Abstract
Models that have the capacity of recognizing the subtle similarities or differences among a set of samples are crucial in many areas of Machine Learning. We present a novel neural model built with attention and recurrence that learns to compare the characteristics of set of objects. Our basic model outperforms strong baselines based on Deep ConvNets in many challenging visual tasks. We tested the generalization capacity of this model by using it for one shot classification on the Omniglot dataset, where it showed comparable results with other methods.

## Usage
1. Install dependencies
    * Scipy and Numpy
    * [Theano](http://deeplearning.net/software/theano/) `pip install --upgrade https://github.com/Theano/Theano/archive/master.zip`
    * [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`

2. Download this repository and prepare the data. Running the following scripts will download, preprocess and store the dataset on the disk for use during training. From the home directory of the repo:
  * To prepare Omniglot dataset: `cd data; python setup_omniglot.py; cd ..;`
  * To prepare LFW dataset: `cd data; python setup_lfw.py; cd ..;`

3. Train the networks. Refer to command line help for specifying by entering `python <script.py> -h`
  * To train Binary ARC model from the paper on Verfication task: `python arc_verif.py`
  * To train Convolutional ARC model from the paper on Verfication task: `python carc_verif.py`
  * To train the 50 layer Wide ResNet baseline on Verfication task: `python wrn_verif.py`

## Pretrained Models
Some of the pretrained models are available [here](https://drive.google.com/drive/folders/0B2EI3F3FJpunenBGZTBPSnZsRVE?usp=sharing)
