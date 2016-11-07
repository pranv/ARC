import numpy as np
from numpy.random import choice, random

import os

EXPT_START_NUM = 100
NUM_EXPTS = 100

for expt_num in range(EXPT_START_NUM, EXPT_START_NUM + NUM_EXPTS):
	arg_string = 'python arc_omniglot_verif.py'

	arg_string += " -n " + str(expt_num) 							# expt-name
	arg_string += " -l " + str(10.0 ** (-4 * random() - 3)) 		# learning-rate
	arg_string += " -i " + str(32)									# image-size
	arg_string += " -a " + str(choice([4, 6])) 						# attn-win
	arg_string += " -s " + str(choice([256, 512, 1024]))			# lstm-states
	arg_string += " -g " + str(choice([8])) 						# glimpses
	arg_string += " -f " + str(0.25) 								# fg-bias
	arg_string += " -b " + str(choice([32])) 						# batch-size
	arg_string += " -p " + str(choice([0.1, 0.3, 0.5])) 			# dropout
	arg_string += " -m " + str(20000) 								# max-iterations

	os.system(arg_string)
