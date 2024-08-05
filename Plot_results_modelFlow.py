import sys, os
import numpy as np
# define the function
import matplotlib.pyplot as plt
import matplotlib as mpl
import keras
# save np.load
np_load_old = np.load
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras import models, layers, utils, backend as K
from sklearn import preprocessing
from glob import glob
np.set_printoptions(threshold=sys.maxsize)

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

cmap = plt.get_cmap('gray_r')
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

model_dir='trained_modelFlow/'

history_cnn = np.load(model_dir+'training_histories.npz')['arr_0']
#print(history_cnn)
model_cnn = keras.models.load_model("trained_modelFlow/cnn.h5",custom_objects={"F1": F1 })
print(model_cnn)

N = 0;
#todo: shared with train model, make a function
outdir = sys.argv[1]
for fn in glob(outdir+"*.npz")[:100]:
	l = np.load(fn);print('loading file', l)
	data = l['images'];print('images.shape', data.shape)
	obs = l['flow_data'];print('flowdata.shape', obs.shape)
	#print('We have {} events and {} true v2 '.format(data.shape[0],obs.shape[0]))
	N += data.shape[0];

	ndim = 3*32*32
	#ndim = 1*32*32
	x = data.reshape(data.shape[0],ndim);
	preprocessing.normalize(x,norm='l2',copy=False);
	y_pred = model_cnn.predict(x);
	try:
		Y_pred = np.concatenate((Y_pred, y_pred));
		y = np.concatenate((y,obs));
	except NameError:
		Y_pred = y_pred;
		y = obs;

	#print("Loaded {}".format(fn));

#y = preprocessing.normalize(y,norm='l1');
y = y[:,0] # v2 only
print(y)
#print(y_test[:5],"\n",predictions_cnn[:,0][:5]);

fig,ax = plt.subplots(1,1,figsize=(7,5));
#ax.scatter(y_test,predictions_cnn[:,0]);
#z,xe,ye = np.histogram2d(y_test,predictions_cnn[:,0],bins=(100,100));#,weights=myevent['pt'])
#m = z.max();
##z[z < 1] = np.nnan
#x = 0.5*(xe[:-1]+xe[1:]);
#y = 0.5*(ye[:-1]+ye[1:]);
#ax.contourf(x,y,z,levels=10,norm=mpl.colors.LogNorm(1,m));
#print(Y_pred[:,0])
#print(np.nanmin(y)) # nan
print("Prediction minimum is: ",np.min(Y_pred))
#print(np.nanmax(y)) # nan
print("Prediction maximum is: ",np.max(Y_pred))
xlims = (np.min(y), np.max(y))
h = ax.hist2d(y,Y_pred[:,0],bins=100,range=(xlims, xlims), cmap="RdBu_r");
ax.set_xlabel("$v_2^\\mathrm{true}$");
ax.set_ylabel("$v_2^\\mathrm{pred}$");
fig.colorbar(h[3],ax=ax);
line = np.linspace(xlims[0], xlims[1], 100)
plt.plot(line,line,'g')

#fig.tight_layout();
fig.savefig('figs/modelFlow_corr.pdf',bbox_inches="tight");#,dpi=150);

