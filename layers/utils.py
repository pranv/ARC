def ortho_init(shape):
	"""
	taken from: https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py#L327-L367
	"""
	a = np.random.normal(0.0, 1.0, shape)
	u, _, v = np.linalg.svd(a, full_matrices=False)
	W = u if u.shape == shape else v 	# pick the one with the correct shape
	return W.astype(dtype)


def normal_init(shape, sigma):
	W = np.random.normal(0.0, sigma, shape)
	return W.astype(dtype)


def batched_dot(A, B):
	C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])      
	return C.sum(axis=-2)
