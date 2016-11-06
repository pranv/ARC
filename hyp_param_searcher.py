import numpy as np
from numpy.random import choice, random

import os

EXPT_START_NUM = 0
NUM_EXPTS = 100

for expt_num in range(EXPT_START_NUM, EXPT_START_NUM + NUM_EXPTS):
	arg_string = 'python arc_omniglot_verif.py'

	arg_string += " -n " + str(expt_num) 							# expt-name
	arg_string += " -l " + str(10.0 ** (-4 * random() - 2)) 		# learning-rate
	arg_string += " -i " + str(32)									# image-size
	arg_string += " -a " + str(choice([4, 6, 8])) 					# attn-win
	arg_string += " -s " + str(choice([256, 512, 1024, 2048]))		# lstm-states
	arg_string += " -g " + str(choice([8, 16])) 					# glimpses
	arg_string += " -f " + str(choice([0.25, 0.5, 1.0, 2.0])) 		# fg-bias
	arg_string += " -b " + str(choice([32, 64])) 					# batch-size
	arg_string += " -p " + str(choice([0.0, 0.1, 0.3, 0.5])) 		# dropout
	arg_string += " -m " + str(5000) 								# max-iterations

	os.system(arg_string)
