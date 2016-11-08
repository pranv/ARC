import argparse
parser = argparse.ArgumentParser(description="Command Line Interface for Setting HyperParameter Values")
parser.add_argument("-n", "--expt-name", type=str, default="a_o_test", help="experiment name for logging purposes")
parser.add_argument("-l", "--learning-rate", type=float, default=3e-5, help="global leaning rate")
parser.add_argument("-i", "--image-size", type=int, default=32, help="size of the square input image (side)")
parser.add_argument("-a", "--attn-win", type=int, default=4, help="size of square attention window (side)")
parser.add_argument("-s", "--lstm-states", type=int, default=512, help="number of LSTM controller states")
parser.add_argument("-g", "--glimpses", type=int, default=8, help="number of glimpses per image")
parser.add_argument("-f", "--fg-bias", type=float, default=0.25, help="initial bias of the forget gate of LSTM controller")
parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size for training")
parser.add_argument("-t", "--testing", action="store_true", help="report test set results")
parser.add_argument("-m", "--max-iter", type=int, default=300000, help="number of iteration to train the net for")
parser.add_argument("-p", "--dropout", type=float, default=0.0, help="dropout on the input")
parser.add_argument("-u", "--hyp-tuning", action="store_false", help="add conditional terminations while tuning params")

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
dropout = meta_data["dropout"]
N_ITER_MAX = meta_data["max_iter"]

data_split = [30, 10]
val_freq = 1000
val_num_batches = 500
test_num_batches = 2000


print "... importing libraries"
import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_all_params, get_output
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam
from lasagne.layers import helper

from layers import ARC

from data_workers import Omniglot

import time
import cPickle
import gzip


print "... setting up the network"
X = T.tensor3("input")
y = T.imatrix("target")

l_in = InputLayer(shape=(None, image_size, image_size), input_var=X)
l_noise = DropoutLayer(l_in, p=dropout)
l_arc = ARC(l_noise, lstm_states=lstm_states, image_size=image_size, attn_win=attn_win, 
					glimpses=glimpses, fg_bias_init=fg_bias)
l_dense = DenseLayer(l_arc, lstm_states/2)
l_y = DenseLayer(l_dense, 1, nonlinearity=sigmoid)

prediction = get_output(l_y)
prediction_clean = get_output(l_y, deterministic=True)

loss = T.mean(binary_crossentropy(prediction, y))
accuracy = T.mean(T.eq(prediction_clean > 0.5, y), dtype=theano.config.floatX)

params = get_all_params(l_y)
updates = adam(loss, params, learning_rate=learning_rate)

meta_data["num_param"] = lasagne.layers.count_params(l_y)
print "number of parameters: ", meta_data["num_param"]

print "... compiling"
train_fn = theano.function([X, y], outputs=loss, updates=updates)
val_fn = theano.function([X, y], outputs=[loss, accuracy])

print "... loading dataset"
worker = Omniglot(img_size=image_size, data_split=data_split)

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
