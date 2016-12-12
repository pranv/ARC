num_trials = 128
predictor_file = 'ARC_OS.opf'
embedder_file = None
embedding_size = 512


import numpy as np

from main import deserialize
from data_workers import OmniglotOSLake, OmniglotVinyals


if embedder_file is None:
    predictor = deserialize(predictor_file)
else:
    def predictor(X):
        predictor = deserialize(predictor_file)
        embedder = deserialize(embedder_file)
        embeddings = embedder(X).reshape(-1, 20, embedding_size)
        return predictor(embeddings)


worker = OmniglotOSLake()
X_OS, t_OS = worker.fetch_batch()

all_acc = []
for run in range(20):
	X = X_OS[run]
	t = t_OS[run]

	y = predictor(X).reshape(20, 20).argmax(axis=1)
	run_acc = np.mean(y == t) * 100.0
	print "run ", run + 1, ": ", run_acc 
	all_acc.append(run_acc)

print "accuracy: ", np.mean(all_acc), "%"


worker = OmniglotVinyals()

all_acc = []
for run in range(20):
	X, t = worker.fetch_batch()

	y = predictor(X).reshape(20, 20).argmax(axis=1)
	run_acc = np.mean(y == t) * 100.0
	print "run ", run + 1, ": ", run_acc 
	all_acc.append(run_acc)

print "accuracy: ", np.mean(all_acc), "%"
