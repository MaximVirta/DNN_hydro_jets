import keras
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

def generate_batches(files, batch_size):
	ind = 0;
	while True:
		fname = files[ind];

		ind = (ind + 1) % len(files);

		print(ind)
		io = np.load(fname);

		x = io["images"];
		y = io["flow_data"];

		ndim = 3*32*32
		x = x.reshape(x.shape[0],ndim)
		x = preprocessing.normalize(x,norm='l2',copy=False);
		y = y[:,0] # v2 only

		#y = preprocessing.quantile_transform(y)

		for l in range(0, x.shape[0], batch_size):
			in_local = x[l:(l+batch_size)]; # Input is 3x32x32 histograms.
			out_local = y[l:(l+batch_size)]; # Output is v2.

			yield in_local, out_local

def getDataset(files, batch_size):
	ds = tf.data.Dataset.from_generator(
		generator=lambda: generate_batches(files=files, batch_size=batch_size),
		output_types = (tf.float32, tf.float32),
		output_shapes = ([None, 3, 32, 32], [None, 1])
	)

	return ds
