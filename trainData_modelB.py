import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras import models, layers, utils, backend as K
from plot_mymodel import *
from model_B import *
from glob import glob

from datasetGenerator_modelB import *

outdir = sys.argv[1];

files = []

for i in range(50):
    for fn in glob(outdir+str(i)+"/"+"*.npz")[:]:
        #print(fn)
        files.append(fn);

print("Loaded {} files".format(len(files)));

# exec(open("model_B.py").read())

n_features = 3*32*32 # per event[:,0]

model_cnn = MyModel(n_features)

plot_model(model_cnn, to_file='figs/modelB_plot.png', show_shapes=True, show_layer_names=True)
#visualize_nn(model_cnn)

# Compile model
model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        # loss='mean_absolute_error');
#              loss=tf.keras.losses.MeanSquaredError());
        loss=tf.keras.losses.MeanSquaredLogarithmicError());

#epochs=10 to not overfit
#batch_size=32 (PRD)

train_frac = int(0.85*len(files))
train_files = files[:train_frac]
test_files = files[train_frac:]

#print("Training model with {} files and validationg with {} files".format(len(train_files), len(test_files));

train_generator = generate_batches(files=train_files, batch_size=20);
test_generator = generate_batches(files=test_files, batch_size=20);


history_cnn = model_cnn.fit(
	steps_per_epoch=len(train_files),
	#use_multiprocessing=True,
	#workers=6,
	batch_size=128,
	x=train_generator,
	verbose=1,
	max_queue_size=32,
	epochs=5,
	# callbacks=callbacks_list,
	validation_data=test_generator,
	validation_steps=len(test_files)
	)

try:
	model_dir='trained_modelB/'
	os.mkdir(model_dir);
except FileExistsError:
	pass;
model_cnn.save(model_dir+'cnn.h5')
np.savez(model_dir+'training_histories.npz', [ history.history for history in [ history_cnn ]])

