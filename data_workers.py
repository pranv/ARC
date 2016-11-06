import numpy as np

from numpy.random import choice

import theano

from scipy.misc import imresize as resize


class Omniglot(object):
	def __init__(self, path='data/omniglot.npy', img_size=32, data_split=[30, 10]):
		chars = np.load(path)
		chars = chars.astype(theano.config.floatX)
		chars /= 255.0

		# resize the images
		resized_chars = np.zeros((1623, 20, img_size, img_size))
		for i in range(1623):
			for j in range(20):
				resized_chars[i, j] = resize(chars[i, j], (img_size, img_size))
		chars = resized_chars

		# mean subtraction
		chars -= chars.mean()

		# starting index (char) of each alphabet
		a_start = [0, 20, 49, 75, 116, 156, 180, 226, 240, 266, 300, 333, 355, 381,
						  424, 448, 496, 518, 534, 586, 633, 673, 699, 739, 780, 813,
						  827, 869, 892, 909, 964, 984, 1010, 1036, 1062, 1088, 1114,
						  1159, 1204, 1245, 1271, 1318, 1358, 1388, 1433, 1479, 1507,
						  1530, 1555, 1597]

		# size of each alphabet (num of chars)
		a_size = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
							16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
							26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

		# slicing indices for splitting data
		i = a_start[data_split[0]] + a_size[data_split[0]]
		j = a_start[data_split[0] + data_split[1]] + a_size[data_split[1]]

		# slicing indices for splitting a_start & a_size
		i = data_split[0]
		j = data_split[0] + data_split[1]
		starts = {}
		starts['train'], starts['val'], starts['test'] = a_start[:i], a_start[i:j], a_start[j:]
		sizes = {}
		sizes['train'], sizes['val'], sizes['test']  = a_size[:i], a_size[i:j], a_size[j:]
		
		def size2p(size):
			s = np.array(size).astype('float64')
			return s / s.sum()

		p = {}
		p['train'], p['val'], p['test'] = size2p(sizes['train']), size2p(sizes['val']), size2p(sizes['test'])

		self.data = chars
		self.starts = starts
		self.sizes = sizes
		self.p = p
		self.img_size = img_size

	def fetch_verif_batch(self, batch_size, part='train'):
		"""
			This outputs batch_size number of pairs
			Thus the actual number of images outputted is 2 * batch_size
			The Batch is divided into 4 parts:
				Dissimilar A 		Dissimilar B
				Similar A 			Similar B

			Corresponding images in Similar A and Similar B form the similar pair
			similarly, Dissimilar A and Dissimilar B form the dissimilar pair
			
			When flattened, the batch has 4 parts with indices:
				Dissimilar A 		0 - batch_size / 2
				Similar A    		batch_size / 2  - batch_size
				Dissimilar B 		batch_size  - 3 * batch_size / 2
				Similar B 			3 * batch_size / 2 - batch_size

		"""
		data = self.data
		starts = self.starts[part]
		sizes = self.sizes[part]
		p = self.p[part]
		img_size = self.img_size

		num_alphbts = len(starts)

		X = np.zeros((2 * batch_size, img_size, img_size), dtype=theano.config.floatX)
		for i in range(batch_size/2):
			# sampling dissimilar pairs
			alphbt_idx = choice(num_alphbts, p=p)			# select a alphabet
			char_offset = choice(sizes[alphbt_idx], 2, replace=False)	# select 2 distinct chars, by selecting its offset within the alphabet
			char_idxs = starts[alphbt_idx] + char_offset		# calculate char index	
			X[i], X[i+batch_size] = data[char_idxs, choice(20, 2, replace=False)] # choose 2 drawers

			# sampling similar pairs
			alphbt_idx = choice(num_alphbts, p=p)			# select an alphabet
			char_offset = choice(sizes[alphbt_idx]) 	# select a char, by selecting its offset within the alphabet
			char_idx = starts[alphbt_idx] + char_offset	# calculate char index
			X[i+batch_size/2], X[i+3*batch_size/2] = data[char_idx, choice(20, 2, replace=False)] # choose 2 drawers

		y = np.zeros((batch_size, 1), dtype='int32')
		y[:batch_size/2] = 0
		y[batch_size/2:] = 1

		return X, y
