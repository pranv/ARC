import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_all_params, get_output
from lasagne.objectives import binary_crossentropy, binary_accuracy
from lasagne.updates import adam
from lasagne.layers import helper

from layers import SimpleARC
from data_workers import OmniglotOS
from main import train, test, deserialize


def create_embedder_fn(glimpses):
    X = T.tensor4("input")
    l_in = InputLayer(shape=(None, 1, 32, 32), input_var=X)
    l_arc = SimpleARC(l_in, lstm_states=512, image_size=32, attn_win=4, 
                    glimpses=glimpses, fg_bias_init=0.0)
    embedding = get_output(l_arc, deterministic=True)
    embedding_fn = theano.function([X], outputs=embedding)
    
    params = deserialize('ARC_OS' + '.params')
    helper.set_all_param_values(l_arc, params[:2])
    
    return embedding_fn


worker = OmniglotOS(image_size=32, batch_size=128)

X_test, y_test = worker.fetch_batch('test')

for glimpses in range(1, 9):
    embedding_fn = create_embedder_fn(glimpses)

    X = T.matrix("embedding")
    y = T.imatrix("target")
    l_in = InputLayer(shape=(None, 512), input_var=X)
    l_y = DenseLayer(l_in, 1, nonlinearity=sigmoid)
    prediction = get_output(l_y)
    loss = T.mean(binary_crossentropy(prediction, y))
    accuracy = T.mean(binary_accuracy(prediction, y))
    params = get_all_params(l_y)
    updates = adam(loss, params, learning_rate=1e-3)
    train_fn = theano.function([X, y], outputs=loss, updates=updates)
    val_fn = theano.function([X, y], outputs=[loss, accuracy])

    for i in range(2000):
        X_train, y_train = worker.fetch_batch('train')
        print train_fn(embedding_fn(X_train), y_train)

    X_train, y_train = worker.fetch_batch('train')
    train_loss = train_fn(embedding_fn(X_train), y_train)
    val_loss, val_acc = val_fn(embedding_fn(X_test), y_test)
    print "number of glimpses per image: ", glimpses
    print "\ttraining performance:"
    print "\t\t loss:", train_loss
    print "\ttesting performance:" 
    print "\t\t loss:", val_loss
    print "\t\t accuracy:", val_acc

np.save('X_test', X_test)
np.save('y_test', y_test)
