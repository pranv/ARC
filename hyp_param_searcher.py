import numpy as np
from numpy.random import choice, random

import os

EXPT_START_NUM = 3000
NUM_EXPTS = 1000

for expt_num in range(EXPT_START_NUM, EXPT_START_NUM + NUM_EXPTS):
	arg_string = 'python arc_omniglot_verif.py'

	arg_string += " -n " + str(expt_num) 							# expt-name
	arg_string += " -l " + str(10.0 ** (-3 * random() - 3)) 		# learning-rate
	arg_string += " -i " + str(32)									# image-size
	arg_string += " -a " + str(choice([4, 6])) 						# attn-win
	arg_string += " -s " + str(choice([256, 512, 1024]))			# lstm-states
	
	# glimpses
	g = choice([8, 16, 32])											
	arg_string += " -g " + str(g) 
	
	# forget gate bias init
	if g == 8:
		fb = 0.25
	elif g == 16:
		fb = 1.0
	elif g == 32:
		fb = 2.0
	arg_string += " -f " + str(fb) 								# fg-bias
	
	arg_string += " -b " + str(choice([32, 64])) 					# batch-size
	arg_string += " -p " + str(choice([0.0, 0.1, 0.3, 0.5])) 	# dropout
	arg_string += " -m " + str(100000) 								# max-iterations
	arg_string += " -h"

	os.system(arg_string)
