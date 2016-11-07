import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.gradient import grad_clip

import lasagne


PI = np.pi
dtype = theano.config.floatX


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


class ARC(lasagne.layers.Layer):
	def __init__(self, incoming, lstm_states, image_size, attn_win, 
					glimpses, fg_bias_init, learnt_params=None, final_state_only=True, **kwargs):
		super(ARC, self).__init__(incoming, **kwargs)
		num_input = attn_win ** 2

		if learnt_params is None:
			# W_lstm is the whole weight matrix of the LSTM controller.
			# takes in (num_input + lstm_states + 1, B) input to give (4 * lstm_states, B) output
			W_lstm = np.zeros((4 * lstm_states, num_input + lstm_states + 1), dtype=dtype)
			for i in range(4):
				W_lstm[i*lstm_states:(i + 1)*lstm_states, :num_input] = ortho_init(shape=(lstm_states, num_input))
				W_lstm[i*lstm_states:(i + 1)*lstm_states, num_input:-1] = ortho_init(shape=(lstm_states, lstm_states))
			W_lstm[2*lstm_states:3*lstm_states, -1] = fg_bias_init
			W_g = normal_init(shape=(4, lstm_states), sigma=0.01)
		else:
			W_lstm, W_g = learnt_params

		self.W_lstm = self.add_param(W_lstm, (4 * lstm_states, num_input + lstm_states + 1), name='W_lstm')
		self.W_g = self.add_param(W_g, (4, lstm_states), name='W_g')

		self.lstm_states = lstm_states
		self.image_size = image_size
		self.attn_win = attn_win
		self.glimpses = glimpses
		self.final_state_only = final_state_only

	def attend(self, I, H, W):
		attn_win = self.attn_win
		image_size = self.image_size

		gp = T.tanh(T.dot(W, H.T).T)

		center_y = gp[:, 0].dimshuffle(0, 'x')
		center_x = gp[:, 1].dimshuffle(0, 'x')
		delta = 1.0 - T.abs_(gp[:, 2]).dimshuffle(0, 'x')
		#gamma = T.exp(1.0 - 2 * T.abs_(gp[:, 2])).dimshuffle([0, 'x', 'x'])
		gamma = T.exp(2 * delta - 1.0).dimshuffle([0, 'x', 'x'])

		center_y = image_size * (center_y + 1.0) / 2.0
		center_x = image_size * (center_x + 1.0) / 2.0
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

		G = batched_dot(batched_dot(F_Y, I), F_X.transpose([0, 2, 1]))

		return G

	def get_output_for(self, input, **kwargs):
		B = input.shape[0]		# batch size
		lstm_states = self.lstm_states

		even_input = input[:B/2]
		odd_input = input[B/2:]

		def step(glimpse_count, c_tm1, h_tm1, odd_input, even_input, W_lstm, W_g):
			# c_tm1, h_tm1 are (B/2, lstm_states)
			I = ifelse(T.eq(glimpse_count % 2, 0), even_input, odd_input) # (B/2, image_size, image_size)
			glimpse = self.attend(I, h_tm1, W_g) # (B/2, attn_win, attn_win)
			flat_glimpse = glimpse.reshape((B/2, -1))

			# (4 * lstm_states, num_input + lstm_states + 1) x transpose(B / 2, num_input + lstm_states + 1)
			lstm_ip = T.concatenate([flat_glimpse, h_tm1, T.ones((B/2, 1))], axis=1)
			pre_activation = T.dot(W_lstm, lstm_ip.T) # (4 * lstm_states, B / 2)

			z = T.tanh(pre_activation[0*lstm_states:1*lstm_states])
			i = T.nnet.sigmoid(pre_activation[1*lstm_states:2*lstm_states])
			f = T.nnet.sigmoid(pre_activation[2*lstm_states:3*lstm_states])
			o = T.nnet.sigmoid(pre_activation[3*lstm_states:4*lstm_states])

			c_t = f * c_tm1.T + i * z 	# (lstm_states, B / 2)
			h_t = o * T.tanh(c_t)

			c_t = grad_clip(c_t, -1.0, 1.0)
			h_t = grad_clip(h_t, -1.0, 1.0)

			return glimpse_count + 1, c_t.T, h_t.T # (B/2, lstm_states)

		glimpse_count_0 = 0
		c_0 = T.zeros((B/2, lstm_states))
		h_0 = T.zeros((B/2, lstm_states))

		_, cells, hiddens = theano.scan(fn=step, non_sequences=[odd_input, even_input, self.W_lstm, self.W_g], 
						outputs_info=[glimpse_count_0, c_0, h_0], n_steps=self.glimpses * 2, strict=True)[0]

		if self.final_state_only:
			return hiddens[-1]
		else:
			return hiddens

	def get_output_shape_for(self, input_shape):
		if self.final_state_only:
			# should be: return (input_shape[0]/2, self.lstm_states)
			return (input_shape[0], self.lstm_states) 
		else:
			# should be: return (2 * self.num_glimpses, input_shape[0]/2, self.lstm_states)
			return (2 * self.num_glimpses, input_shape[0], self.lstm_states)



if __name__ == "__main__":
	from lasagne.layers import InputLayer, DenseLayer, get_output
	
	l_in = InputLayer(shape=(None, 32, 32))
	l_arc = ARC(l_in, lstm_states=100, image_size=32, attn_win=4, 
					glimpses=4, fg_bias_init=0.0)
	l_y = DenseLayer(l_arc, 1)
	prediction = get_output(l_y)

	fn = theano.function([l_in.input_var], outputs=prediction)
	X = np.random.randn(4, 32, 32).astype(dtype)
	print fn(X)