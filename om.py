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
