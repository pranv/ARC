import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

import lasagne

dtype = theano.config.floatX
PI = np.pi


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


class BaseARC(lasagne.layers.Layer):
	def __init__(self, incoming, num_input, num_glimpse_params, lstm_states, image_size, attn_win, 
					glimpses, fg_bias_init, final_state_only=True, **kwargs):
		super(BaseARC, self).__init__(incoming, **kwargs)

		W_lstm = np.zeros((4 * lstm_states, num_input + lstm_states + 1), dtype=dtype)
		for i in range(4):
			W_lstm[i*lstm_states:(i+1)*lstm_states, :num_input] = ortho_init(shape=(lstm_states, num_input))
			W_lstm[i*lstm_states:(i+1)*lstm_states, num_input:-1] = ortho_init(shape=(lstm_states, lstm_states))
		W_lstm[2*lstm_states:3*lstm_states, -1] = fg_bias_init
		
		W_g = normal_init(shape=(num_glimpse_params, lstm_states), sigma=0.01)

		self.W_lstm = self.add_param(W_lstm, (4 * lstm_states, num_input + lstm_states + 1), name='W_lstm')
		self.W_g = self.add_param(W_g, (num_glimpse_params, lstm_states), name='W_g')

		self.lstm_states = lstm_states
		self.image_size = image_size
		self.attn_win = attn_win
		self.glimpses = glimpses
		self.final_state_only = final_state_only

	def get_filterbanks(self, gp):
		attn_win = self.attn_win
		image_size = self.image_size

		center_y = gp[:, 0].dimshuffle(0, 'x')
		center_x = gp[:, 1].dimshuffle(0, 'x')
		delta = 1.0 - T.abs_(gp[:, 2]).dimshuffle(0, 'x')
		gamma = T.exp(1.0 - 2 * T.abs_(gp[:, 2])).dimshuffle([0, 'x', 'x'])

		center_y = (image_size - 1) * (center_y + 1.0) / 2.0
		center_x = (image_size - 1) * (center_x + 1.0) / 2.0
		delta = image_size / attn_win * delta

		rng = T.arange(attn_win, dtype=dtype) - attn_win / 2.0 + 0.5
		cX = center_x + delta * rng
		cY = center_y + delta * rng

		a = T.arange(image_size, dtype=dtype)
		b = T.arange(image_size, dtype=dtype)

		F_X = 1.0 + ((a - cX.dimshuffle([0, 1, 'x'])) / gamma) ** 2.0 
		F_Y = 1.0 + ((b - cY.dimshuffle([0, 1, 'x'])) / gamma) ** 2.0
		F_X = 1.0 / (PI * gamma * F_X)
		F_Y = 1.0 / (PI * gamma * F_Y)
		F_X = F_X / (F_X.sum(axis=-1).dimshuffle(0, 1, 'x') + 1e-4)
		F_Y = F_Y / (F_Y.sum(axis=-1).dimshuffle(0, 1, 'x') + 1e-4)

		return F_X, F_Y

	def attend(self, I, H, W):
		raise NotImplementedError('This method must be implemented by subclassed layers')

	def get_output_for(self, input, **kwargs):
		# input is 4D tensor: (batch_size, num_filters, 0, 1)
		image_size = self.image_size
		lstm_states = self.lstm_states
		attn_win = self.attn_win

		B = input.shape[0] / 2 	# number of pairs in batch
		odd_input = input[:B]
		even_input = input[B:]

		def step(glimpse_count, c_tm1, h_tm1, odd_input, even_input, W_lstm, W_g):
			turn = T.eq(glimpse_count % 2, 0)
			I = ifelse(turn, even_input, odd_input)
			
			glimpse = self.attend(I, h_tm1, W_g) 	# (B, attn_win, attn_win)
			flat_glimpse = glimpse.reshape((B, -1))

			lstm_ip = T.concatenate([flat_glimpse, h_tm1, T.ones((B, 1))], axis=1) # (B, num_input + states + 1)
			pre_activation = T.dot(W_lstm, lstm_ip.T) # result: (4 * states, B)

			z = T.tanh(pre_activation[0*lstm_states:1*lstm_states])
			i = T.nnet.sigmoid(pre_activation[1*lstm_states:2*lstm_states])
			f = T.nnet.sigmoid(pre_activation[2*lstm_states:3*lstm_states])
			o = T.nnet.sigmoid(pre_activation[3*lstm_states:4*lstm_states])

			c_t = f * c_tm1.T + i * z 	# all in (states, B)
			h_t = o * T.tanh(c_t)

			return glimpse_count + 1, c_t.T, h_t.T 	# c, h in (B, states)

		glimpse_count_0 = 0
		c_0 = T.zeros((B, lstm_states))
		h_0 = T.zeros((B, lstm_states))

		_, cells, hiddens = theano.scan(fn=step, non_sequences=[odd_input, even_input, self.W_lstm, self.W_g], 
						outputs_info=[glimpse_count_0, c_0, h_0], n_steps=self.glimpses * 2)[0]

		if self.final_state_only:
			return hiddens[-1]
		else:
			return hiddens

	def get_output_shape_for(self, input_shape):
		# the batch size in both must be input_shape[0] / 2
		# but since that it is None, we leave it as it is
		if self.final_state_only:
			return (input_shape[0], self.lstm_states) 
		else:
			return (2 * self.num_glimpses, input_shape[0], self.lstm_states)


class SimpleARC(BaseARC):
	def __init__(self, incoming, lstm_states, image_size, attn_win, glimpses, \
		fg_bias_init, final_state_only=True, **kwargs):

		BaseARC.__init__(self, incoming, attn_win**2, 3, lstm_states, image_size, \
			attn_win, glimpses, fg_bias_init, final_state_only=True, **kwargs)

	def attend(self, I, H, W):
		I = I[:, 0]
		gp = T.dot(W, H.T).T
		F_X, F_Y = self.get_filterbanks(gp)
		G = batched_dot(batched_dot(F_Y, I), F_X.transpose([0, 2, 1]))
		return G


class ConvARC(BaseARC):
	def __init__(self, incoming, num_filters, lstm_states, image_size, attn_win, glimpses, \
		fg_bias_init, final_state_only=True, **kwargs):

		self.num_filters = num_filters

		BaseARC.__init__(self, incoming, num_filters * attn_win ** 2, 3, lstm_states, \
			image_size, attn_win, glimpses, fg_bias_init, final_state_only=True, **kwargs)

	def attend(self, I, H, W):
		I = I[:, 0]
		num_filters = self.num_filters
		gp = T.dot(W, H.T).T
		F_X, F_Y = self.get_filterbanks(gp)
		F_X = F_X.repeat(num_filters, axis=0)
		F_Y = F_Y.repeat(num_filters, axis=0)
		G = batched_dot(batched_dot(F_Y, I), F_X.transpose([0, 2, 1]))
		return G


class ConvARC3DA(BaseARC):
	def __init__(self, incoming, num_filters, lstm_states, image_size, attn_win, glimpses, \
		fg_bias_init, final_state_only=True, **kwargs):

		self.num_filters = num_filters

		BaseARC.__init__(self, incoming, attn_win ** 2, 5, lstm_states, image_size, \
			attn_win, glimpses, fg_bias_init, final_state_only=True, **kwargs)

	def attend(self, I, H, W):
		num_filters = self.num_filters

		gp = T.dot(W, H.T).T

		center_z = gp[:, 2].dimshuffle(0, 'x')
		gamma_z = T.exp(1.0 - 4 * T.abs_(gp[:, 4])).dimshuffle([0, 'x'])
		center_z = (num_filters - 1) * (center_z + 1.0) / 2.0

		c = np.arange(num_filters, dtype=dtype)
		F_Z = 1.0 + ((c - center_z) / gamma_z) ** 2
		F_Z = 1.0 / (PI * gamma_z * F_Z)
		F_Z = F_Z / (F_Z.sum(axis=1).dimshuffle(0, 'x') + 1e-4)
		FM = I * F_Z[:, :, np.newaxis, np.newaxis]
		FM = FM.sum(axis=1)
		
		F_X, F_Y = self.get_filterbanks(gp)
		G = batched_dot(batched_dot(F_Y, FM), F_X.transpose([0, 2, 1]))
		return G
