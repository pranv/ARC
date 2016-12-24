import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_output, helper

from layers import SimpleARC
from data_workers import OmniglotOS
from main import deserialize

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.ion()
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (16, 8)


expt_name = "ARC_OS" 
image_size = 32
attn_win = 4
glimpses = 8
lstm_states = 512
batch_size = 2

X = T.tensor4("input")
y = T.imatrix("target")

l_in = InputLayer(shape=(None, 1, image_size, image_size), input_var=X)
l_arc = SimpleARC(l_in, lstm_states=lstm_states, image_size=image_size, attn_win=attn_win, glimpses=glimpses, fg_bias_init=0.0, final_state_only=False)

embeddings = get_output(l_arc, deterministic=True)

GPs = []
for i in range(-1, 2 * glimpses - 1):
    if i == -1:
        gp = T.dot(l_arc.W_g, T.zeros_like(embeddings[0].T)).T
    else: 
        gp = T.dot(l_arc.W_g, embeddings[i].T).T

    center_y = gp[:, 0].dimshuffle([0, 'x'])
    center_x = gp[:, 1].dimshuffle([0, 'x'])
    delta = 1.0 - T.abs_(gp[:, 2]).dimshuffle([0, 'x'])
    gamma = T.exp(1.0 - 2 * T.abs_(gp[:, 2])).dimshuffle([0, 'x', 'x'])

    center_y = (image_size - 1) * (center_y + 1.0) / 2.0
    center_x = (image_size - 1) * (center_x + 1.0) / 2.0
    delta = image_size / attn_win * delta
    
    GPs.extend([center_y, center_x, delta])
    
embedding_fn = theano.function([X], outputs=GPs)

params = deserialize(expt_name + '.params')
helper.set_all_param_values(l_arc, params[:2])

worker = OmniglotOS(image_size=image_size, batch_size=batch_size)

while(1):
    X_sample, _ = worker.fetch_batch('val')
    G = embedding_fn(X_sample)

    G = np.array(G)
    G = G.reshape(2 * glimpses, 3, batch_size)

    g = G[:, :, 0]
    I1 = X_sample[0, 0]
    I2 = X_sample[2, 0]

    fig_axs = plt.subplots(2, glimpses)
    fig = fig_axs[0]
    axs = fig_axs[1:]
    axs = axs[0]

    for i in range(glimpses):
        ax = axs[0, i]
        ax.imshow(I1, cmap="Greys_r")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.grid(False) 
        ax.yaxis.grid(False) 
        ax.set_title(str(i + 1))
        x, y, w = g[2*i]
        w *= attn_win
        x = x - w / 2.0
        y = 32 - y - w / 2.0
        rect = patches.Rectangle((x, y), w, w, linewidth=(2*w-1)/8, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        ax = axs[1, i]
        ax.imshow(I2, cmap="Greys_r")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.grid(False) 
        ax.yaxis.grid(False)
        x, y, w = g[2*i + 1]
        w *= attn_win
        x = x - w / 2.0
        y = 32 - y - w / 2.0
        rect = patches.Rectangle((x, y), w, w, linewidth=(2*w-1)/8, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    g = G[:, :, 1]
    I1 = X_sample[1, 0]
    I2 = X_sample[3, 0]

    fig_axs = plt.subplots(2, glimpses)
    fig = fig_axs[0]
    plt.subplots_adjust(wspace=0, hspace=0)
    axs = fig_axs[1:]
    axs = axs[0]

    for i in range(glimpses):
        ax = axs[0, i]
        ax.imshow(I1, cmap="Greys_r")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.grid(False) 
        ax.yaxis.grid(False)
        ax.set_title(str(i + 1))
        x, y, w = g[2*i]
        w *= attn_win
        x = x - w / 2.0
        y = 32 - y - w / 2.0
        rect = patches.Rectangle((x, y), w, w, linewidth=(2*w-1)/8, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        ax = axs[1, i]
        ax.imshow(I2, cmap="Greys_r")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.grid(False) 
        ax.yaxis.grid(False)
        x, y, w = g[2*i + 1]
        w *= attn_win
        x = x - w / 2.0
        y = 32 - y - w / 2.0
        rect = patches.Rectangle((x, y), w, w, linewidth=(2*w-1)/8, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    raw_input()
