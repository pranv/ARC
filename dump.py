class OmniglotOSNaive(Omniglot):
	def __init__(self, path='data/omniglot.npy', image_size=32, data_split=[20, 10], within_alphabet=True):
		Omniglot.__init__(self, path, 20, image_size, data_split, within_alphabet)
		
		def size2p(size):
			s = (np.array(size) >= 20) * np.array(size).astype('float64')
			return s / s.sum()

		sizes = self.sizes
		p = {}
		p['train'], p['val'], p['test'] = size2p(sizes['train']), size2p(sizes['val']), size2p(sizes['test'])

		self.p = p

	def fetch_batch(self, part):
		data = self.data
		starts = self.starts[part]
		sizes = self.sizes[part]
		p = self.p[part]
		image_size = self.image_size
		within_alphabet = self.within_alphabet

		num_alphbts = len(starts)

		X = np.zeros((20 * 40, 1, image_size, image_size))
		y = np.zeros((20), dtype='int32')
		
		for alphabet in xrange(20):
			if within_alphabet:
				char_offsets = choice(sizes[alphabet], 20, replace=False)
				char_idxs = starts[alphabet] + char_offsets
			else:
				char_idxs = choice(range(starts[0], starts[-1] + sizes[-1]), 20, replace=False)
			
			key = choice(20)
			key_idx = char_idxs[key]

			T = np.zeros((2 * 20, image_size, image_size), dtype='uint8')
			T[:20] = data[char_idxs, choice(20)]
			T[20:] = data[key_idx, choice(20)]
			
			T = T / 255.0
			T = T - self.mean_pixel
			T = T[:, np.newaxis]
			T = T.astype(theano.config.floatX)

			k = alphabet * 20
			X[k:k+20] = T[:20]
			k = 20 * 20 + alphabet * 20
			X[k:k+20] = T[20:]
			y[alphabet] = key

		X = X.astype(theano.config.floatX)
		
		return X, y


class OmniglotOSLake(object):
	def __init__(self, image_size=32):
		X = np.load('data/one_shot/X.npy')
		y = np.load('data/one_shot/y.npy')

		# resize the images
		resized_X = np.zeros((20, 800, image_size, image_size), dtype='uint8')
		for i in xrange(20):
			for j in xrange(800):
				resized_X[i, j] = resize(X[i, j], (image_size, image_size))
		X = resized_X

		self.mean_pixel = 0.08 # dataset mean pixel

		self.X = X
		self.y = y

	def fetch_batch(self):
		X = self.X
		y = self.y

		X = X / 255.0
		X = X - self.mean_pixel
		X = X[:, :, np.newaxis]
		X = X.astype(theano.config.floatX)

		y = y.astype('int32')

		return X, y

NAIVE OS

import numpy as np

from main import train, test, serialize, deserialize
from data_workers import OmniglotOSNaive


print "... loading data"
worker = OmniglotOSNaive(within_alphabet=True, data_split=[20, 10])

print "... testing"
predictor = deserialize('ARC_VERIF_omniglot_standard.opf')

acc = 0.0
for _ in range(100):
	X, t = worker.fetch_batch('test')
	y = predictor(X)
	y = y.reshape(20, 20).argmax(axis=1)
	acc += (np.sum(y == t) / 20.0)
	print _, acc / (_ + 1)

print "accuracy: ", acc * 100. / 100



ONE SHOT (OM)


"""
uncomment all lines to see plots of each item in each run
"""

import numpy as np

#import matplotlib.pyplot as plt
#plt.ion()
#plt.style.use('fivethirtyeight')
#plt.rcParams["figure.figsize"] = (10, 8)

#import os
#os.mkdir('one_shot_results')

from data_workers import OmniglotOSLake
from main import deserialize

print "... loading data"
worker = OmniglotOSLake()
X_OS, t_OS = worker.fetch_batch()

print "... loading model"
#predictor = deserialize('WRN_VERIF_omniglot_small.opf')
predictor = deserialize('ConvARC_VERIF_omniglot_standard_deep_attn.opf')
#predictor = deserialize('ARC_VERIF_omniglot_standard.opf')

#fig, axes = plt.subplots(nrows=4, ncols=5)

print "... testing"

all_acc = []
for run in range(20):
	X = X_OS[run]
	t = t_OS[run]

	y = predictor(X).reshape(20, 20).argmax(axis=1)
	run_acc = np.mean(y == t) * 100.0
	print "run ", run + 1, ": ", run_acc 
	all_acc.append(run_acc)

	#for i in range(20):
	#	fig.suptitle('run: ' + str(run+1) + '    item: ' +  str(i+1) + '\nprediction: ' + str(y[i]+1) + '    truth: ' + str(t[i]+1))
	#	for j in range(20):
	#		axes.flat[j].matshow(np.concatenate([X[i*20+j, 0], X[i*20+400+j, 0]], axis=1))
	#		axes.flat[j].set_xticks([])
	#		axes.flat[j].set_yticks([])
	#		axes.flat[j].set_title(str(j + 1))
	#	plt.show()
	#	plt.savefig('one_shot_results/' + 'run' + str(run+1) + '_item' +  str(i+1))

print "accuracy: ", np.mean(all_acc), "%"

