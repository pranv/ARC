import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

import lasagne

from utils import normal_init, ortho_init, batched_dot


PI = np.pi
dtype = theano.config.floatX


class ConvARC(lasagne.layers.Layer):
	def __init__(self, incoming, num_filters, lstm_states, image_size, attn_win, 
					glimpses, fg_bias_init, final_state_only=True, **kwargs):
		super(ConvARC, self).__init__(incoming, **kwargs)
		
		num_input = num_filters * (attn_win ** 2)

		W_lstm = np.zeros((4 * lstm_states, num_input + lstm_states + 1), dtype=dtype)
		for i in range(4):
			W_lstm[i*lstm_states:(i + 1)*lstm_states, :num_input] = ortho_init(shape=(lstm_states, num_input))
			W_lstm[i*lstm_states:(i + 1)*lstm_states, num_input:-1] = ortho_init(shape=(lstm_states, lstm_states))
		W_lstm[2*lstm_states:3*lstm_states, -1] = fg_bias_init
		W_g = normal_init(shape=(3, lstm_states), sigma=0.01)

		self.W_lstm = self.add_param(W_lstm, (4 * lstm_states, num_input + lstm_states + 1), name='W_lstm')
		self.W_g = self.add_param(W_g, (3, lstm_states), name='W_g')

		self.num_filters = num_filters
		self.lstm_states = lstm_states
		self.image_size = image_size
		self.attn_win = attn_win
		self.glimpses = glimpses
		self.final_state_only = final_state_only

	def attend(self, I, H, W):
		attn_win = self.attn_win
		image_size = self.image_size
		num_filters = self.num_filters

		gp = T.dot(W, H.T).T

		center_y = gp[:, 0].dimshuffle(0, 'x')
		center_x = gp[:, 1].dimshuffle(0, 'x')
		delta = 1.0 - T.abs_(gp[:, 2]).dimshuffle(0, 'x')
		gamma = T.exp(1.0 - 2 * T.abs_(gp[:, 2])).dimshuffle([0, 'x', 'x'])

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

		F_X = F_X.repeat(num_filters, axis=0)
		F_Y = F_Y.repeat(num_filters, axis=0)

		G = batched_dot(batched_dot(F_Y, I), F_X.transpose([0, 2, 1]))

		return G

	def get_output_for(self, input, **kwargs):
		
		image_size = self.image_size
		num_filters = self.num_filters
		lstm_states = self.lstm_states
		attn_win = self.attn_win

		# input is 4D tensor: (batch_size, num_filters, 0, 1)
		B = input.shape[0] / 2 		# pairs in batch
		odd_input = input[:B]
		even_input = input[B:]

		# (B * num_filters, image_size, image_size)
		odd_input = odd_input.reshape((B * num_filters, image_size, image_size))
		even_input = even_input.reshape((B * num_filters, image_size, image_size))

		def step(glimpse_count, c_tm1, h_tm1, odd_input, even_input, W_lstm, W_g):
			# c_tm1, h_tm1 are (B, lstm_states)
			
			turn = T.eq(glimpse_count % 2, 0)
			I = ifelse(turn, even_input, odd_input)
			
			# (B, attn_win, attn_win)
			glimpse = self.attend(I, h_tm1, W_g)
			flat_glimpse = glimpse.reshape((B, num_filters * (attn_win ** 2)))

			# (4 * states, num_input + states + 1) x transpose(B, num_input + states + 1)
			# result: (4 * states, B)
			lstm_ip = T.concatenate([flat_glimpse, h_tm1, T.ones((B, 1))], axis=1)
			pre_activation = T.dot(W_lstm, lstm_ip.T) 	

			z = T.tanh(pre_activation[0*lstm_states:1*lstm_states])
			i = T.nnet.sigmoid(pre_activation[1*lstm_states:2*lstm_states])
			f = T.nnet.sigmoid(pre_activation[2*lstm_states:3*lstm_states])
			o = T.nnet.sigmoid(pre_activation[3*lstm_states:4*lstm_states])

			# all in (states, B)
			c_t = f * c_tm1.T + i * z
			h_t = o * T.tanh(c_t)

			#c_t = T.clip(c_t, -1.0, 1.0)
			#h_t = T.clip(h_t, -1.0, 1.0)

			# output: (B, states)
			return glimpse_count + 1, c_t.T, h_t.T

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
		# but since that it is none, we leave it as it is
		if self.final_state_only:
			return (input_shape[0], self.lstm_states) 
		else:
			return (2 * self.num_glimpses, input_shape[0], self.lstm_states)


if __name__ == "__main__":
	
	from lasagne.layers import InputLayer, Conv2DLayer, get_output
	
	l_in = InputLayer(shape=(2, 1, 7, 7))
	l_conv = Conv2DLayer(l_in, num_filters=1, filter_size=(3, 3), stride=(1, 1), pad='same')
	l_carc = ConvARC(l_conv, num_filters=1, lstm_states=5, image_size=7, \
						attn_win=2, glimpses=1, fg_bias_init=0.0)

	y = get_output(l_carc)

	fn = theano.function([l_in.input_var], outputs=y)
	X = np.random.random((2, 1, 7, 7)).astype(dtype)
	
	print fn(X).shape
