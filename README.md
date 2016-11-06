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
  * To prepare Omniglot dataset: `python data/setup_omniglot.py`
  * To prepare LFW dataset: `python data/setup_lfw.py`

3. Train the networks. The following code segments train the models with the default hyper-parameters. The hyper-paramters can however be changed by passing command line arguments. Refer to command line help for this by entering `python <script.py> -h`
  * To train Binary ARC model from the paper on Omniglot Verfication task: `python arc_omniglot_verif.py`
  * To train the 50 layer Wide ResNet baseline on Omniglot Verfication task: `python wrn_omniglot_verif.py`
