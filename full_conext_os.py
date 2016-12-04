import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, LSTMLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, ExpressionLayer, ElemwiseSumLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.init import HeNormal
from lasagne.layers import get_all_params, get_output
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.updates import adam
from lasagne.layers import helper

from main import train, test, serialize, deserialize
from data_workers import OmniglotOS


num_states = 128
embedding_size = 512

print "... setting up the network"
X = T.tensor3("input")
y = T.ivector("target")

l_in = InputLayer(shape=(None, 20, embedding_size), input_var=X)
l_lstm_up = LSTMLayer(l_in, num_states, learn_init=True, grad_clipping=5., )
l_lstm_down = LSTMLayer(l_in, num_states, learn_init=True, grad_clipping=5., backwards=True)
l_merge = ConcatLayer([l_lstm_up, l_lstm_down])
l_rshp1 = ReshapeLayer(l_merge, (-1, 2 * num_states))
l_dense = DenseLayer(l_rshp1, 1, W=HeNormal(gain='relu'), nonlinearity=rectify)
l_rshp2 = ReshapeLayer(l_dense, (-1, 20))
l_y = NonlinearityLayer(l_rshp2, softmax)

prediction = get_output(l_y)

loss = T.mean(categorical_crossentropy(prediction, y))
accuracy = T.mean(categorical_accuracy(prediction, y))

params = get_all_params(l_y, trainable=True)
updates = adam(loss, params, learning_rate=1e-3)

print "... compiling"
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
val_fn = theano.function(inputs=[X, y], outputs=[loss, accuracy])

print "... loading data"
worker = OmniglotOS('ARC_VERIF_omniglot_standard', embedding_size, within_alphabet=True, num_trails=128, data_split=[30, 10])

meta_data = {}
meta_data["n_iter"] = 50000
meta_data["num_output"] = 20

meta_data, params = train(train_fn, val_fn, worker, meta_data, get_params=lambda: helper.get_all_param_values(l_y))
meta_data = test(val_fn, worker, meta_data)
