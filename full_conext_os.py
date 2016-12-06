import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, LSTMLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, ExpressionLayer, ElemwiseSumLayer
from lasagne.nonlinearities import tanh, rectify, elu, softmax
from lasagne.init import HeNormal
from lasagne.layers import get_all_params, get_all_layers, get_output
from lasagne.regularization import regularize_layer_params
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.updates import adam
from lasagne.layers import helper

from main import train, test, serialize, deserialize
from data_workers import OmniglotOSFC


num_states = 64
embedding_size = 256

expt_name = "ConvARC_FCOS_ACROSS"

print "... setting up the network"
X = T.tensor3("input")
y = T.ivector("target")

l_in = InputLayer(shape=(None, 20, embedding_size), input_var=X)


gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
	W_hid=lasagne.init.Orthogonal(), b=lasagne.init.Constant(0.))

forget_gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
	W_hid=lasagne.init.Orthogonal(), b=lasagne.init.Constant(1.))

cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
	W_hid=lasagne.init.Orthogonal(), b=lasagne.init.Constant(0.),
	W_cell=None, nonlinearity=lasagne.nonlinearities.tanh)

l_lstm_up = LSTMLayer(l_in, num_states,
							ingate=gate_parameters, forgetgate=forget_gate_parameters,
							cell=cell_parameters, outgate=gate_parameters, 
							learn_init=True, grad_clipping=100.0)
l_lstm_down = LSTMLayer(l_in, num_states, backwards=True, 
						ingate=gate_parameters, forgetgate=forget_gate_parameters,
						cell=cell_parameters, outgate=gate_parameters,
						learn_init=True, grad_clipping=100.0)


l_merge = ConcatLayer([l_lstm_up, l_lstm_down])
l_rshp1 = ReshapeLayer(l_merge, (-1, 2 * num_states))
l_dense = DenseLayer(l_rshp1, 1, W=HeNormal(gain='relu'), nonlinearity=elu)
l_rshp2 = ReshapeLayer(l_dense, (-1, 20))
l_y = NonlinearityLayer(l_rshp2, softmax)

prediction = get_output(l_y)

loss = T.mean(categorical_crossentropy(prediction, y))
accuracy = T.mean(categorical_accuracy(prediction, y))

params = get_all_params(l_y, trainable=True)
updates = adam(loss, params, learning_rate=3e-4)

print "... compiling"
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
val_fn = theano.function(inputs=[X, y], outputs=[loss, accuracy])
op_fn = theano.function([X], outputs=prediction)

print "... loading data",
worker = OmniglotOSFC('ConvARC_VERIF_omniglot_standard_deep_attn', embedding_size, within_alphabet=False, num_trails=32, data_split=[30, 10])
print "within_alphabet:", worker.within_alphabet

meta_data = {}
meta_data["n_iter"] = 25000
meta_data["num_output"] = 20

meta_data, params = train(train_fn, val_fn, worker, meta_data, get_params=lambda: helper.get_all_param_values(l_y))
meta_data = test(val_fn, worker, meta_data)

serialize(params, expt_name + '.params')
serialize(meta_data, expt_name + '.mtd')
serialize(op_fn, expt_name + '.opf')
