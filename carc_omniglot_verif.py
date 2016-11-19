import argparse
parser = argparse.ArgumentParser(description="Command Line Interface for Setting HyperParameter Values")
parser.add_argument("-n", "--expt-name", type=str, default="a_o_test", help="experiment name for logging purposes")
parser.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="global leaning rate")
parser.add_argument("-i", "--image-size", type=int, default=32, help="size of the square input image (side)")
parser.add_argument("-w", "--attn-win", type=int, default=4, help="size of square attention window (side)")
parser.add_argument("-s", "--lstm-states", type=int, default=512, help="number of LSTM controller states")
parser.add_argument("-g", "--glimpses", type=int, default=8, help="number of glimpses per image")
parser.add_argument("-f", "--fg-bias", type=float, default=0.2, help="initial bias of the forget gate of LSTM controller")
parser.add_argument("-b", "--batch-size", type=int, default=128, help="batch size for training")
parser.add_argument("-t", "--testing", action="store_true", help="report test set results")
parser.add_argument("-m", "--max-iter", type=int, default=1000000, help="number of iteration to train the net for")
parser.add_argument("-u", "--hyp-tuning", action="store_true", help="add conditional terminations while tuning params")
parser.add_argument("-d", "--depth", type=int, default=8, help="the resnet has depth equal to 6d+2")
parser.add_argument("-k", "--width", type=int, default=4, help="width multiplier for each WRN block")
parser.add_argument("-a", "--within-alphabet", action="store_false", help="select only the character pairs that within the alphabet ")

meta_data = vars(parser.parse_args())

for md in meta_data.keys():
	print md, meta_data[md]

expt_name = meta_data["expt_name"]
learning_rate = meta_data["learning_rate"]
image_size = meta_data["image_size"]
attn_win = meta_data["attn_win"]
glimpses = meta_data["glimpses"]
lstm_states = meta_data["lstm_states"]
fg_bias = meta_data["fg_bias"]
batch_size = meta_data["batch_size"]
N_ITER_MAX = meta_data["max_iter"]
within_alphabet = meta_data["within_alphabet"]

wrn_n = meta_data["depth"]
wrn_k = meta_data["width"]

data_split = [30, 10]
val_freq = 1000
val_num_batches = 500
test_num_batches = 2000


print "... importing libraries"
import sys
sys.setrecursionlimit(10000)

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
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam
from lasagne.layers import helper

from layers import ConvARC

from data_workers import Omniglot

import time
import cPickle
import gzip


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


print "... setting up the network"
n_filters = {0: 16, 1: 16 * wrn_k, 2: 32 * wrn_k}

X = T.tensor4("input")
y = T.imatrix("target")

l_in = InputLayer(shape=(None, 1, image_size, image_size), input_var=X)

# first layer, output is 16 x 32 x 32
l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3, 3), \
	stride=(1, 1), nonlinearity=rectify, pad='same', W=HeNormal(gain='relu')))

# first stack of residual blocks, output is (16 * wrn_k) x 32 x 32
l = residual_block(l, first=True, filters=n_filters[1])
for _ in range(1, wrn_n):
	l = residual_block(l, filters=n_filters[1])

# second stack of residual blocks, output is (32 * wrn_k) x 16 x 16
l = residual_block(l, increase_dim=True, filters=n_filters[2])
for _ in range(1, (wrn_n+2)):
	l = residual_block(l, filters=n_filters[2])

bn_post_conv = BatchNormLayer(l)
bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

l_carc = ConvARC(bn_post_relu, num_filters=n_filters[2], lstm_states=lstm_states, image_size=16, 
					attn_win=attn_win, glimpses=glimpses, fg_bias_init=fg_bias)
l_y = DenseLayer(l_carc, num_units=1, nonlinearity=sigmoid)

meta_data["num_param"] = lasagne.layers.count_params(l_y)
print "number of parameters: ", meta_data["num_param"]

prediction = get_output(l_y)
prediction_clean = get_output(l_y, deterministic=True)

loss = T.mean(binary_crossentropy(prediction, y))
accuracy = T.mean(T.eq(prediction_clean > 0.5, y), dtype=theano.config.floatX)

all_layers = get_all_layers(l_y)
l2_penalty = 0.0001 * regularize_layer_params(all_layers, lasagne.regularization.l2)
loss = loss + l2_penalty

params = get_all_params(l_y, trainable=True)
updates = adam(loss, params, learning_rate=learning_rate)

print "... compiling"
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
val_fn = theano.function(inputs=[X, y], outputs=[loss, accuracy])

print "... loading dataset"
worker = Omniglot(image_size=image_size, data_split=data_split, within_alphabet=within_alphabet)

print "... begin training"
meta_data["training_loss"] = []
meta_data["validation_loss"] = []
meta_data["validation_accuracy"] = []

best_val_loss = np.inf
best_val_acc = 0.0
iter_n = 0
best_iter_n = 0
best_params = helper.get_all_param_values(l_y)

smooth_loss = 0.6932
try:
	while iter_n < N_ITER_MAX:
		iter_n += 1

		tick = time.clock()
		X_train, y_train = worker.fetch_verif_batch(batch_size, 'train')
		X_train = X_train.reshape(-1, 1, image_size, image_size)
		batch_loss = train_fn(X_train, y_train)
		tock = time.clock()
		meta_data["training_loss"].append((iter_n, batch_loss))

		smooth_loss = 0.99 * smooth_loss + 0.01 * batch_loss
		print "iteration: ", iter_n, " | ", np.round((tock - tick), 3) * 1000, "ms", " | training loss: ", np.round(smooth_loss, 3)
		
		if np.isnan(batch_loss):
			print "... NaN Detected, terminating"
			break

		if meta_data['hyp_tuning']:
			if smooth_loss > 0.3 and iter_n > 80000:
				print "... poor performace, terminating"
				break

			if smooth_loss > 0.4 and iter_n > 40000:
				print "... poor performace, terminating"
				break

			if smooth_loss > 0.5 and iter_n > 20000:
				print "... poor performace, terminating"
				break

			if smooth_loss > 0.6 and iter_n > 10000:
				print "... poor performace, terminating"
				break

			if smooth_loss > 0.65 and iter_n > 5000:
				print "... poor performace, terminating"
				break

			if smooth_loss > 0.69 and iter_n > 2500:
				print "... poor performace, terminating"
				break


		if iter_n % val_freq == 0:
			net_val_loss, net_val_acc = 0.0, 0.0
			for i in range(val_num_batches):
				X_val, y_val = worker.fetch_verif_batch(batch_size, 'val')
				X_val = X_val.reshape(-1, 1, image_size, image_size)
				val_loss, val_acc = val_fn(X_val, y_val)
				net_val_loss += val_loss
				net_val_acc += val_acc
			val_loss = net_val_loss / val_num_batches
			val_acc = net_val_acc / val_num_batches

			print "****" * 20
			print "validation loss: ", val_loss
			print "validation accuracy: ", val_acc * 100.0
			print "****" * 20

			meta_data["validation_loss"].append((iter_n, val_loss))
			meta_data["validation_accuracy"].append((iter_n, val_acc))

			if val_acc > best_val_acc:
				best_val_acc = val_acc

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_iter_n = iter_n
				best_params = helper.get_all_param_values(l_y)

except KeyboardInterrupt:
	pass

print "... training done"
print "best validation accuracy: ", best_val_acc * 100.0, " at iteration number: ", best_iter_n

if meta_data["testing"]:
	"... setting up testing network"
	helper.set_all_param_values(l_y, best_params)
	net_test_loss, net_test_acc = 0.0, 0.0
	for i in range(test_num_batches):
		X_test, y_test = worker.fetch_verif_batch(batch_size, 'test')
		X_test = X_test.reshape(-1, 1, image_size, image_size)
		test_loss, test_acc = val_fn(X_test, y_test)
		net_test_loss += test_loss
		net_test_acc += test_acc
	test_loss = net_test_loss / test_num_batches
	test_acc = net_test_acc / test_num_batches

	print "====" * 20
	print "final testing loss: ", test_loss
	print "final testing accuracy: ", test_acc * 100.0
	print "====" * 20

	meta_data["testing_loss"] = test_loss
	meta_data["testing_accuracy"] = test_acc

print "... serializing metadata"
log_md = gzip.open("results/" + str(expt_name) + ".mtd", "wb")
cPickle.dump(meta_data, log_md)
log_md.close()

print "... serializing parameters"
log_p = gzip.open("results/" + str(expt_name) + ".params", "wb")
cPickle.dump(best_params, log_p)
log_p.close()

print "... exiting ..."
