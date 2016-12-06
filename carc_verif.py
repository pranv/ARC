import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import batch_norm, BatchNormLayer, ExpressionLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer, GlobalPoolLayer
from lasagne.nonlinearities import rectify, sigmoid
from lasagne.init import HeNormal
from lasagne.layers import get_all_params, get_all_layers, get_output
from lasagne.regularization import regularize_layer_params
from lasagne.objectives import binary_crossentropy, binary_accuracy
from lasagne.updates import adam
from lasagne.layers import helper

from layers import ConvARC3DA
from data_workers import OmniglotVerif
from main import train, test, serialize, deserialize

import sys
sys.setrecursionlimit(10000)

import argparse


def residual_block(l, increase_dim=False, projection=True, first=False, filters=16):
	if increase_dim:
		first_stride = (2, 2)
	else:
		first_stride = (1, 1)
	
	if first:
		bn_pre_relu = l
	else:
		bn_pre_conv = BatchNormLayer(l)
		bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)
	
	conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=HeNormal(gain='relu')))
	dropout = DropoutLayer(conv_1, p=0.3)
	conv_2 = ConvLayer(dropout, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=HeNormal(gain='relu'))
	
	if increase_dim:
		projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
		block = ElemwiseSumLayer([conv_2, projection])
	elif first:
		projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
		block = ElemwiseSumLayer([conv_2, projection])
	else:
		block = ElemwiseSumLayer([conv_2, l])
	
	return block


parser = argparse.ArgumentParser(description="CLI for specifying hyper-parameters")
parser.add_argument("-n", "--expt-name", type=str, default="", help="experiment name(for logging purposes)")
parser.add_argument("--dataset", type=str, default="omniglot", help="omniglot/LFW")

parser.add_argument("--learning-rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--image-size", type=int, default=32, help="side length of the square input image")

parser.add_argument("--attn-win", type=int, default=4, help="side length of square attention window")
parser.add_argument("--lstm-states", type=int, default=256, help="number of LSTM controller states")
parser.add_argument("--glimpses", type=int, default=8, help="number of glimpses per image")
parser.add_argument("--fg-bias-init", type=float, default=0.2, help="initial bias for the forget gate of LSTM controller")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout on the input")
parser.add_argument("--reload-weights", action="store_true", default=False, help="use pretrained weights")

parser.add_argument("--within-alphabet", action="store_false", help="select only the character pairs that within the alphabet ")
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--testing", action="store_true", help="report test set results")
parser.add_argument("--n-iter", type=int, default=200000, help="number of iterations")

parser.add_argument("--wrn-depth", type=int, default=3, help="the resnet has depth equal to 4d+7")
parser.add_argument("--wrn-width", type=int, default=2, help="width multiplier for each WRN block")

meta_data = vars(parser.parse_args())
meta_data["expt_name"] = "ConvARC3DA_VERIF_" + meta_data["dataset"] + "_" + meta_data["expt_name"]

for md in meta_data.keys():
	print md, meta_data[md]

expt_name = meta_data["expt_name"]
learning_rate = meta_data["learning_rate"]
image_size = meta_data["image_size"]
attn_win = meta_data["attn_win"]
glimpses = meta_data["glimpses"]
lstm_states = meta_data["lstm_states"]
fg_bias_init = meta_data["fg_bias_init"]
batch_size = meta_data["batch_size"]
n_iter = meta_data["n_iter"]
within_alphabet = meta_data["within_alphabet"]
wrn_n = meta_data["wrn_depth"]
wrn_k = meta_data["wrn_width"]
data_split = [30, 10]
meta_data["num_output"] = 2

print "... setting up the network"
n_filters = {0: 16, 1: 16 * wrn_k, 2: 32 * wrn_k}

X = T.tensor4("input")
y = T.imatrix("target")

l_in = InputLayer(shape=(None, 1, image_size, image_size), input_var=X)

# first layer, output is 16 x 32 x 32 | (1)
l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3, 3), \
	stride=(1, 1), nonlinearity=rectify, pad='same', W=HeNormal(gain='relu')))

# first stack of residual blocks, output is (16 * wrn_k) x 32 x 32 | (3 + 2 * (n - 1))
l = residual_block(l, first=True, filters=n_filters[1])
for _ in range(1, wrn_n):
	l = residual_block(l, filters=n_filters[1])

# second stack of residual blocks, output is (32 * wrn_k) x 16 x 16 | (3 + 2 * (n + 1))
l = residual_block(l, increase_dim=True, filters=n_filters[2])
for _ in range(1, (wrn_n+2)):
	l = residual_block(l, filters=n_filters[2])

bn_post_conv = BatchNormLayer(l)
bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

l_carc = ConvARC3DA(bn_post_relu, num_filters=n_filters[2], lstm_states=lstm_states, image_size=16, 
					attn_win=attn_win, glimpses=glimpses, fg_bias_init=fg_bias_init)
l_y = DenseLayer(l_carc, num_units=1, nonlinearity=sigmoid)

prediction = get_output(l_y)
prediction_clean = get_output(l_y, deterministic=True)
embedding = get_output(l_carc, deterministic=True)

loss = T.mean(binary_crossentropy(prediction, y))
accuracy = T.mean(binary_accuracy(prediction_clean, y))

all_layers = get_all_layers(l_y)
l2_penalty = 0.0001 * regularize_layer_params(all_layers, lasagne.regularization.l2)
loss = loss + l2_penalty

params = get_all_params(l_y, trainable=True)
updates = adam(loss, params, learning_rate=learning_rate)

meta_data["num_param"] = lasagne.layers.count_params(l_y)
print "number of parameters: ", meta_data["num_param"]

print "... compiling"
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
val_fn = theano.function(inputs=[X, y], outputs=[loss, accuracy])
embed_fn = theano.function([X], outputs=embedding)
op_fn = theano.function([X], outputs=prediction_clean)

if meta_data["reload_weights"]:
	print "... loading pretrained weights"
	params = deserialize(expt_name + '.params')
	helper.set_all_param_values(l_y, params)

print "... loading dataset"
if meta_data["dataset"] == "omniglot":
	worker = OmniglotVerif(image_size=image_size, batch_size=batch_size, \
		data_split=data_split, within_alphabet=within_alphabet)

meta_data, best_params = train(train_fn, val_fn, worker, meta_data, \
	get_params=lambda: helper.get_all_param_values(l_y))

if meta_data["testing"]:
	print "... testing"
	helper.set_all_param_values(l_y, best_params)
	meta_data = test(val_fn, worker, meta_data)

serialize(params, expt_name + '.params')
serialize(meta_data, expt_name + '.mtd')
serialize(embed_fn, expt_name + '.emf')
serialize(op_fn, expt_name + '.opf')
