import numpy as np
from numpy.random import choice

import theano

from scipy.misc import imresize as resize

from image_augmenter import ImageAugmenter

from main import serialize, deserialize


class Omniglot(object):
	def __init__(self, path='data/omniglot.npy', batch_size=128, image_size=32):
		"""
		path: path to omniglot.npy file produced by "data/setup_omniglot.py" script
		batch_size: the output is (2 * batch size, 1, image_size, image_size)
					X[i] & X[i + batch_size] are the pair
		image_size: size of the image
		data_split: in number of alphabets, e.g. [30, 10] means out of 50 Omniglot characters, 
					30 is for training, 10 for validation and the remaining(10) for testing
		within_alphabet: for verfication task, when 2 characters are sampled to form a pair, 
						this flag specifies if should they be from the same alphabet/language
		---------------------
		Data Augmentation Parameters:
			flip: here flipping both the images in a pair
			scale: x would scale image by + or - x%
			rotation_deg
			shear_deg
			translation_px: in both x and y directions
		"""
		chars = np.load(path)

		# resize the images
		resized_chars = np.zeros((1623, 20, image_size, image_size), dtype='uint8')
		for i in xrange(1623):
			for j in xrange(20):
				resized_chars[i, j] = resize(chars[i, j], (image_size, image_size))
		chars = resized_chars

		self.mean_pixel = chars.mean() / 255.0	# used later for mean subtraction

		# starting index of each alphabet in a list of chars
		a_start = [0, 20, 49, 75, 116, 156, 180, 226, 240, 266, 300, 333, 355, 381,
						  424, 448, 496, 518, 534, 586, 633, 673, 699, 739, 780, 813,
						  827, 869, 892, 909, 964, 984, 1010, 1036, 1062, 1088, 1114,
						  1159, 1204, 1245, 1271, 1318, 1358, 1388, 1433, 1479, 1507,
						  1530, 1555, 1597]

		# size of each alphabet (num of chars)
		a_size = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
							16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
							26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

		# each alphabet/language has different number of characters.
		# in order to uniformly sample all characters, we need weigh the probability 
		# of sampling a alphabet by its size. p is that probability
		def size2p(size):
			s = np.array(size).astype('float64')
			return s / s.sum()

		self.size2p = size2p

		self.data = chars
		self.a_start = a_start
		self.a_size = a_size
		self.image_size = image_size
		self.batch_size = batch_size

		flip = True
		scale = 0.2
		rotation_deg = 20
		shear_deg = 10
		translation_px = 5
		self.augmentor = ImageAugmenter(image_size, image_size,
                 hflip=flip, vflip=flip,
                 scale_to_percent=1.0+scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                 translation_x_px=translation_px, translation_y_px=translation_px)

	def fetch_batch(self, part):
		"""
			This outputs batch_size number of pairs
			Thus the actual number of images outputted is 2 * batch_size
			Say A & B form the half of a pair
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
		pass


class OmniglotVerif(Omniglot):
	def __init__(self, path='data/omniglot.npy', batch_size=128, image_size=32):
		Omniglot.__init__(self, path, batch_size, image_size)

		a_start = self.a_start
		a_size = self.a_size

		# slicing indices for splitting a_start & a_size
		i = 30
		j = 40
		starts = {}
		starts['train'], starts['val'], starts['test'] = a_start[:i], a_start[i:j], a_start[j:]
		sizes = {}
		sizes['train'], sizes['val'], sizes['test']  = a_size[:i], a_size[i:j], a_size[j:]
		
		size2p = self.size2p

		p = {}
		p['train'], p['val'], p['test'] = size2p(sizes['train']), size2p(sizes['val']), size2p(sizes['test'])

		self.starts = starts
		self.sizes = sizes
		self.p = p

	def fetch_batch(self, part):
		data = self.data
		starts = self.starts[part]
		sizes = self.sizes[part]
		p = self.p[part]
		image_size = self.image_size
		batch_size = self.batch_size

		num_alphbts = len(starts)

		X = np.zeros((2 * batch_size, image_size, image_size), dtype='uint8')
		for i in xrange(batch_size / 2):
			# choose similar chars
			same_idx = choice(range(starts[0], starts[-1] + sizes[-1]))

			# choose dissimilar chars within alphabet
			alphbt_idx = choice(num_alphbts, p=p)
			char_offset = choice(sizes[alphbt_idx], 2, replace=False)
			diff_idx = starts[alphbt_idx] + char_offset
			
			X[i], X[i + batch_size] = data[diff_idx, choice(20, 2)]
			X[i + batch_size / 2], X[i + 3 * batch_size / 2] = data[same_idx, choice(20, 2, replace=False)]	
		
		y = np.zeros((batch_size, 1), dtype='int32')
		y[:batch_size / 2] = 0
		y[batch_size / 2:] = 1

		if part == 'train':
			X = self.augmentor.augment_batch(X)
		else:
			X = X / 255.0
		
		X = X - self.mean_pixel
		X = X[:, np.newaxis]
		X = X.astype(theano.config.floatX)

		return X, y


class OmniglotOS(Omniglot):
	def __init__(self, path='data/omniglot.npy', batch_size=128, image_size=32):
		Omniglot.__init__(self, path, batch_size, image_size)

		a_start = self.a_start
		a_size = self.a_size

		num_train_chars = a_start[29] + a_size[29]

		train = self.data[:num_train_chars, :16]	# (964, 16, H, W)
		val = self.data[:num_train_chars, 16:] 	# (964, 4, H, W)
		test = self.data[num_train_chars:] 	# (659, 20, H, W)

		# slicing indices for splitting a_start & a_size
		i = 30
		starts = {}
		starts['train'], starts['val'], starts['test'] = a_start[:i], a_start[:i], a_start[i:]
		sizes = {}
		sizes['train'], sizes['val'], sizes['test']  = a_size[:i], a_size[:i], a_size[i:]

		size2p = self.size2p

		p = {}
		p['train'], p['val'], p['test'] = size2p(sizes['train']), size2p(sizes['val']), size2p(sizes['test'])
		
		data = {}
		data['train'], data['val'], data['test'] = train, val, test

		num_drawers = {}
		num_drawers['train'], num_drawers['val'], num_drawers['test'] = 16, 4, 20

		self.data = data
		self.starts = starts
		self.sizes = sizes
		self.p = p
		self.num_drawers = num_drawers

	def fetch_batch(self, part):
		data = self.data[part]
		starts = self.starts[part]
		sizes = self.sizes[part]
		p = self.p[part]
		num_drawers = self.num_drawers[part]
		
		image_size = self.image_size
		batch_size = self.batch_size

		num_alphbts = len(starts)

		X = np.zeros((2 * batch_size, image_size, image_size), dtype='uint8')
		for i in xrange(batch_size / 2):
			# choose similar chars
			same_idx = choice(range(data.shape[0]))

			# choose dissimilar chars within alphabet
			alphbt_idx = choice(num_alphbts, p=p)
			char_offset = choice(sizes[alphbt_idx], 2, replace=False)
			diff_idx = starts[alphbt_idx] + char_offset - starts[0]

			X[i], X[i + batch_size] = data[diff_idx, choice(num_drawers, 2)]
			X[i + batch_size / 2], X[i + 3 * batch_size / 2] = data[same_idx, choice(num_drawers, 2, replace=False)]	
		
		y = np.zeros((batch_size, 1), dtype='int32')
		y[:batch_size / 2] = 0
		y[batch_size / 2:] = 1

		if part == 'train':
			X = self.augmentor.augment_batch(X)
		else:
			X = X / 255.0
		
		X = X - self.mean_pixel
		X = X[:, np.newaxis]
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

		self.mean_pixel = 0.0805 # dataset mean pixel

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


class OmniglotVinyals(Omniglot):
	def __init__(self, path='data/omniglot.npy', num_trials=128, image_size=32):
		Omniglot.__init__(self, path, 0, image_size)
		del self.batch_size
		self.num_trials = num_trials

	def fetch_batch(self):
		data = self.data
		image_size = self.image_size
		num_trials = self.num_trials

		X = np.zeros((num_trials * 40, image_size, image_size), dtype='uint8')
		y = np.zeros(num_trials, dtype='int32')
        
		for t in range(num_trials):
			trial = np.zeros((2 * 20, image_size, image_size), dtype='uint8')
			char_choices = range(1200, 1623) # set of all possible chars
			key_char_idx = choice(char_choices) # this will be the char to be matched

			# sample 19 other chars excluding key
			char_choices.remove(key_char_idx)
			other_char_idxs = choice(char_choices, 19, replace=False)

			pos = range(20)
			key_char_pos = choice(pos) # position of the key char out of 20 pairs
			pos.remove(key_char_pos)
			other_char_pos = np.array(pos, dtype='int32')

			drawers = choice(20, 2, replace=False)
			trial[key_char_pos] = data[key_char_idx, drawers[0]]
			trial[other_char_pos] = data[other_char_idxs, drawers[0]]  
			trial[20:] = data[key_char_idx, drawers[1]]

			k = t * 20
			X[k:k+20] = trial[:20]
			k = k + num_trials * 20
			X[k:k+20] = trial[20:]

			y[t] = key_char_pos

		X = X / 255.0
		X = X - self.mean_pixel
		X = X[:, np.newaxis]
		X = X.astype(theano.config.floatX)

		return X, y


class LFWVerif(object):
	def __init__(self, batch_size = 128, split=[80, 10], image_size=64):
		faces = np.load('data/LFW/faces.npy')
		counts = np.load('data/LFW/counts.npy')

		num_people = len(counts)
		num_train = int(np.round(split[0] / 100.0 * num_people))
		num_val = int(np.round(split[1] / 100.0 * num_people))
		num_test = num_people - num_train - num_val

		i = num_train
		j = i + num_val
		k = j + num_test

		num_faces_so_far = np.cumsum(counts)
		train_faces = faces[:num_faces_so_far[i]]
		val_faces = faces[num_faces_so_far[i]:num_faces_so_far[j]]
		test_faces = faces[num_faces_so_far[j]:]

		self.mean_pixel = faces.mean() / 255.0

		train_counts = counts[:i]
		val_counts = counts[i:j]
		test_counts = counts[j:]

		faces = {}
		faces['train'], faces['val'], faces['test'] = train_faces, val_faces, test_faces

		counts = {}
		counts['train'], counts['val'], counts['test'] = train_counts, val_counts, test_counts

		self.i = i
		self.j = j
		self.k = k

		self.faces = faces
		self.counts = counts
		self.batch_size = batch_size

		vflip = False
		hflip = True
		scale = 0.2
		rotation_deg = 15
		shear_deg = 5
		translation_px = 10
		self.augmentor = ImageAugmenter(image_size, image_size,
		         hflip=hflip, vflip=vflip,
		         scale_to_percent=1.0+scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
		         translation_x_px=translation_px, translation_y_px=translation_px)

    
	def fetch_batch(self, part):
		faces = self.faces[part]
		counts = self.counts[part]
		batch_size = self.batch_size

		num_people = counts.shape[0]
		person_start_idx = np.cumsum(counts) - counts

		X = np.zeros((batch_size * 2, 64, 64), dtype='uint8')
        
		while(1):
			try:
				person_idxs = choice(num_people, batch_size, replace=False)
				face_sub_idxs = np.array([choice(counts[idx]) for idx in person_idxs])
				face_idxs = person_start_idx[person_idxs]
				net_index = face_idxs + face_sub_idxs
				X[:batch_size/2] = faces[net_index[:batch_size/2]]
				X[batch_size:3*batch_size/2] = faces[net_index[-batch_size/2:]]

				# sample similar
				similar_p = np.array(counts >= 2, dtype='float64')
				similar_p /= similar_p.sum()

				person_idxs = choice(num_people, batch_size/2, replace=False, p=similar_p)
				face_sub_idxs = np.array([choice(counts[idx], 2) for idx in person_idxs])
				face_idxs = person_start_idx[person_idxs]
				faces_idxsA = face_idxs + face_sub_idxs[:, 0]
				faces_idxsB = face_idxs + face_sub_idxs[:, 1]
				X[batch_size/2:batch_size] = faces[faces_idxsA]
				X[-batch_size/2:] = faces[faces_idxsB]
				
			except IndexError:
				continue
			break

		y = np.zeros((batch_size, 1), dtype='int32')
		y[:batch_size / 2] = 0
		y[batch_size / 2:] = 1

		if part == 'train':
			X = self.augmentor.augment_batch(X)
		else:
			X = X / 255.0

		X = X - self.mean_pixel
		X = X[:, np.newaxis]
		X = X.astype(theano.config.floatX)

		return X, y
