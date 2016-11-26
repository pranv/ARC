import numpy as np

import time
import gzip
import cPickle


def train(train_fn, val_fn, worker, meta_data, get_params):
	n_iter = meta_data["n_iter"]
	val_freq = 1000
	val_num_batches = 250
	patience = 0.1
	
	meta_data["training_loss"] = []
	meta_data["validation_loss"] = []
	meta_data["validation_accuracy"] = []

	best_val_loss = np.inf
	best_val_acc = 0.0
	best_iter_n = 0
	best_params = get_params()

	print "... training"
	try:
		smooth_loss = np.log(meta_data["num_output"])
		iter_n = 0
		while iter_n < n_iter:
			iter_n += 1
			tick = time.clock()
			X_train, y_train = worker.fetch_batch('train')
			batch_loss = train_fn(X_train, y_train)
			tock = time.clock()
			meta_data["training_loss"].append((iter_n, batch_loss))

			smooth_loss = 0.99 * smooth_loss + 0.01 * batch_loss
			print "iteration: ", iter_n, "  train loss: ", np.round(smooth_loss, 4), "\t", np.round((tock - tick), 3) * 1000, "ms"

			if np.isnan(batch_loss):
				print "... NaN Detected, terminating"
				break

			if iter_n % val_freq == 0:
				net_val_loss, net_val_acc = 0.0, 0.0
				for i in xrange(val_num_batches):
					X_val, y_val = worker.fetch_batch('val')
					val_loss, val_acc = val_fn(X_val, y_val)
					net_val_loss += val_loss
					net_val_acc += val_acc
				val_loss = net_val_loss / val_num_batches
				val_acc = net_val_acc / val_num_batches

				meta_data["validation_loss"].append((iter_n, val_loss))
				meta_data["validation_accuracy"].append((iter_n, val_acc))

				print "====" * 20, "\n", "validation loss: ", val_loss, ", validation accuracy: ", val_acc * 100.0, "\n", "====" * 20

				if val_acc > best_val_acc:
					best_val_acc = val_acc
					best_iter_n = iter_n

				if val_loss < best_val_loss:
					best_val_loss = val_loss
					best_params = get_params()

				if val_loss > best_val_loss + patience:
					break


	except KeyboardInterrupt:
		pass

	print "... training done"
	print "best validation accuracy: ", best_val_acc * 100.0, " at iteration number: ", best_iter_n
	print "... exiting training regime"

	return meta_data, best_params


def test(test_fn, worker, meta_data):
	test_num_batches = 2000

	net_test_loss, net_test_acc = 0.0, 0.0
	for i in range(test_num_batches):
		X_test, y_test = worker.fetch_batch('test')
		test_loss, test_acc = test_fn(X_test, y_test)
		net_test_loss += test_loss
		net_test_acc += test_acc
	test_loss = net_test_loss / test_num_batches
	test_acc = net_test_acc / test_num_batches

	print "====" * 20, "\n", "test loss: ", test_loss, ", test accuracy: ", test_acc * 100.0, "\n", "====" * 20

	meta_data["testing_loss"] = test_loss
	meta_data["testing_accuracy"] = test_acc

	return meta_data


def save(meta_data, params):
	print "... serializing parameters" 
	with gzip.open("results/" + meta_data["expt_name"] + ".params", "wb") as log_p:
		cPickle.dump(params, log_p)
		log_p.close()

	print "... serializing metadata"
	with gzip.open("results/" + meta_data["expt_name"] + ".mtd", "wb") as log_md:
		cPickle.dump(meta_data, log_md)
		log_md.close()

	return
