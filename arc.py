import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_all_params, get_output
from lasagne.objectives import binary_crossentropy, binary_accuracy
from lasagne.updates import adam
from lasagne.layers import helper

from layers import SimpleARC
from data_workers import OmniglotOS
from main import train, test, serialize, deserialize

import argparse


parser = argparse.ArgumentParser(description="CLI for specifying hyper-parameters")
parser.add_argument("-n", "--expt-name", type=str, default="", help="experiment name(for logging purposes)")
parser.add_argument("--dataset", type=str, default="omniglot", help="omniglot/LFW")

meta_data = vars(parser.parse_args())
meta_data["expt_name"] = "ARC_OS_" + meta_data["expt_name"]

for md in meta_data.keys():
	print md, meta_data[md]

expt_name = meta_data["expt_name"]
learning_rate = 1e-4
image_size = 32
attn_win = 4
glimpses = 8
lstm_states = 512
fg_bias_init = 0.2
dropout = 0.2
meta_data["n_iter"] = n_iter = 1500000
batch_size = 128
meta_data["num_output"] = 2

print "... setting up the network"
X = T.tensor4("input")
y = T.imatrix("target")

l_in = InputLayer(shape=(None, 1, image_size, image_size), input_var=X)
l_noise = DropoutLayer(l_in, p=dropout)
l_arc = SimpleARC(l_noise, lstm_states=lstm_states, image_size=image_size, attn_win=attn_win, 
					glimpses=glimpses, fg_bias_init=fg_bias_init)
l_y = DenseLayer(l_arc, 1, nonlinearity=sigmoid)

prediction = get_output(l_y)
prediction_clean = get_output(l_y, deterministic=True)
embedding = get_output(l_arc, deterministic=True)

loss = T.mean(binary_crossentropy(prediction, y))
accuracy = T.mean(binary_accuracy(prediction_clean, y))

params = get_all_params(l_y)
updates = adam(loss, params, learning_rate=learning_rate)

meta_data["num_param"] = lasagne.layers.count_params(l_y)
print "number of parameters: ", meta_data["num_param"]

print "... compiling"
train_fn = theano.function([X, y], outputs=loss, updates=updates)
val_fn = theano.function([X, y], outputs=[loss, accuracy])
embed_fn = theano.function([X], outputs=embedding)
op_fn = theano.function([X], outputs=prediction_clean)

print "... loading dataset"
worker = OmniglotOS(image_size=image_size, batch_size=batch_size)

meta_data, params = train(train_fn, val_fn, worker, meta_data, \
		get_params=lambda: helper.get_all_param_values(l_y))

print "... testing"
helper.set_all_param_values(l_y, params)
meta_data = test(val_fn, worker, meta_data)

serialize(params, expt_name + '.params')
serialize(meta_data, expt_name + '.mtd')
serialize(embed_fn, expt_name + '.emf')
serialize(op_fn, expt_name + '.opf')
