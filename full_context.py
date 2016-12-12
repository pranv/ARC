expt_name = 'ARC_OSFC'
emf = 'ARC_OS.emf'
embedding_size = 512

num_trials = 128
image_size = 32
num_states = 64

import numpy as np
from numpy.random import choice

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer,  NonlinearityLayer
from lasagne.layers import LSTMLayer
from lasagne.layers import ConcatLayer, ReshapeLayer
from lasagne.layers import Gate
from lasagne.nonlinearities import tanh, elu, softmax
from lasagne.init import HeNormal, Orthogonal, Constant
from lasagne.layers import get_all_params, get_output
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.updates import adam
from lasagne.layers import helper

from scipy.misc import imresize as resize
from image_augmenter import ImageAugmenter
from main import train, test, serialize, deserialize

from data_workers import OmniglotOS


def fetch_batch(self, part):
    data = self.data[part]
    starts = self.starts[part]
    sizes = self.sizes[part]
    p = self.p[part]
    num_drawers = self.num_drawers[part]
    image_size = self.image_size
    num_trials = self.num_trials
    num_alphbts = len(starts)

    X = np.zeros((num_trials * 40, image_size, image_size), dtype='uint8')
    y = np.zeros(num_trials, dtype='int32')

    for t in range(num_trials):
        trial = np.zeros((2 * 20, image_size, image_size), dtype='uint8')
        alphbt_idx = choice(num_alphbts) # choose an alphabet
        char_choices = range(sizes[alphbt_idx]) # set of all possible chars
        key_char_idx = choice(char_choices) # this will be the char to be matched

        # sample 19 other chars excluding key
        char_choices.pop(key_char_idx)
        other_char_idxs = choice(char_choices, 19)

        key_char_idx = starts[alphbt_idx] + key_char_idx - starts[0]
        other_char_idxs = starts[alphbt_idx] + other_char_idxs - starts[0]

        pos = range(20)
        key_char_pos = choice(pos) # position of the key char out of 20 pairs
        pos.pop(key_char_pos)
        other_char_pos = np.array(pos, dtype='int32')

        trial[key_char_pos] = data[key_char_idx, choice(num_drawers)]
        trial[other_char_pos] = data[other_char_idxs, choice(num_drawers)]  
        trial[20:] = data[key_char_idx, choice(num_drawers)]

        k = t * 20
        X[k:k+20] = trial[:20]
        k = k + num_trials * 20
        X[k:k+20] = trial[20:]

        y[t] = key_char_pos

    if part == 'train':
        X = self.augmentor.augment_batch(X)
    else:
        X = X / 255.0

    X = X - self.mean_pixel
    X = X[:, np.newaxis]
    X = X.astype(theano.config.floatX)
    
    E = embedding_fn(X)
    E = E.reshape(num_trials, 20, embedding_size)
    return E, y

OmniglotOS.fetch_batch = fetch_batch

worker = OmniglotOS(image_size=image_size)
del worker.batch_size
worker.num_trials = num_trials

embedding_fn = deserialize(emf)


X = T.tensor3("input")
y = T.ivector("target")

l_in = InputLayer(shape=(None, 20, embedding_size), input_var=X)

gate_parameters = Gate(W_in=Orthogonal(), W_hid=Orthogonal(), b=Constant(0.))
forget_gate_parameters = Gate(W_in=Orthogonal(), W_hid=Orthogonal(), b=Constant(1.))
cell_parameters = Gate(W_in=Orthogonal(), W_hid=Orthogonal(), b=Constant(0.), W_cell=None, nonlinearity=tanh)

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

meta_data = {}
meta_data["n_iter"] = 25000
meta_data["num_output"] = 20

meta_data, params = train(train_fn, val_fn, worker, meta_data, get_params=lambda: helper.get_all_param_values(l_y))
meta_data = test(val_fn, worker, meta_data)

serialize(params, expt_name + '.params')
serialize(meta_data, expt_name + '.mtd')
serialize(op_fn, expt_name + '.opf')
