import numpy as np

from main import train, test, serialize, deserialize
from data_workers import OmniglotOSNaive


print "... loading data"
worker = OmniglotOSNaive(within_alphabet=True)

print "... testing"
predictor = deserialize('WRN_VERIF_omniglot_small.opf')

acc = 0.0
for _ in range(20):
	X, t = worker.fetch_batch('test')
	y = predictor(X)
	y = y.reshape(20, 20).argmax(axis=1)
	acc += (np.sum(y == t) / 20.0)

print "accuracy: ", acc * 100. / 20