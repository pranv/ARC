import numpy as np

from main import train, test, serialize, deserialize
from data_workers import OmniglotOSNaive


print "... loading data"
worker = OmniglotOSNaive(within_alphabet=True)

print "... testing"
predictor = deserialize('ConvARC_VERIF_omniglot_standard_deep_attn.opf')

acc = 0.0
for _ in range(100):
	X, t = worker.fetch_batch('test')
	y = predictor(X)
	y = y.reshape(20, 20).argmax(axis=1)
	acc += (np.sum(y == t) / 20.0)
	print _, acc / (_ + 1)

print "accuracy: ", acc * 100. / 100