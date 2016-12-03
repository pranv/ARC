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
from lasagne.updates import rmsprop
from lasagne.layers import helper

from main import train, test, serialize, deserialize
from data_workers import OmniglotOS


print "... setting up the network"
X = T.tensor3("input")
y = T.ivector("target")

l_in = InputLayer(shape=(None, 20, 512), input_var=X)
l_lstm1 = LSTMLayer(l_in, 64)
l_lstm2 = LSTMLayer(l_in, 64)
l_rev2 = ExpressionLayer(l_lstm2, lambda I: I[:,range(19, -1, -1)], output_shape='auto')
l_merge = ConcatLayer([l_lstm1, l_rev2], axis=2)
l_rshp1 = ReshapeLayer(l_merge, (-1, 128))
l_dense = DenseLayer(l_rshp1, 1, W=HeNormal(gain='relu'), nonlinearity=rectify)
l_rshp2 = ReshapeLayer(l_dense, (-1, 20))
l_y = NonlinearityLayer(l_rshp2, softmax)

prediction = get_output(l_y)

loss = T.mean(categorical_crossentropy(prediction, y))
accuracy = T.mean(categorical_accuracy(prediction, y))

params = get_all_params(l_y, trainable=True)
updates = rmsprop(loss, params, learning_rate=1e-2)

print "... compiling"
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
val_fn = theano.function(inputs=[X, y], outputs=[loss, accuracy])

print "... loading data"
worker = OmniglotOS('ARC_VERIF_omniglot_standard', 512, within_alphabet=True, num_trails=32)

meta_data = {}
meta_data["n_iter"] = 100000
meta_data["num_output"] = 20

meta_data, params = train(train_fn, val_fn, worker, meta_data, get_params=lambda: helper.get_all_param_values(l_y))
meta_data = test(val_fn, worker, meta_data)
