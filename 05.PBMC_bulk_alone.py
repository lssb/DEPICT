import tensorflow as tf
import utils
# import functools
import numpy as np
from sklearn import preprocessing
import keras_tuner as kt
from  matplotlib import  pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import os 

import random
seed=4
seed=7


def scale_x(x):
	# 预处理
	# 先对非0特征取log
	# 然后进行归一化
	min_max_scaler = preprocessing.MinMaxScaler()
	standar_scaler = preprocessing.StandardScaler()
	x[x!=0] = np.log(x[x!=0])
	x = standar_scaler.fit_transform(x)
	x = min_max_scaler.fit_transform(x)
	return x

def load_data(path):
	raw_data = np.loadtxt(path, dtype=np.float32, delimiter=",", skiprows=1)
	return scale_x(raw_data[..., :263]), (raw_data[..., 263])


data_dir = "../01.train_data/individual/"
train_x, train_y = load_data(data_dir + "train.csv")
valid_x, valid_y = load_data(data_dir + "valid.csv")
test_x, test_y = load_data(data_dir + "test.csv")




def model_with_seed(seed):
	random.seed(seed)# 为python设置随机种子
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)# 为numpy设置随机种子
	tf.compat.v1.set_random_seed(seed)# tf cpu fix seed
	os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed


	# 263 -> TPM



	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(32, activation="relu"))

	for _ in range(4):
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Dense(32, activation="relu"
			)
		)
	
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(1))

	model.compile(
		loss = "mse",
		optimizer = "adam"
	)



	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

	y_pre_batch = []
	class logPredictPer100epoch(tf.keras.callbacks.Callback):
		def on_epoch_end(self, batch, logs=None):
			example_result = self.model.predict(test_x)
			y_pre_batch.append(tf.reshape(example_result,[-1]))


	history = model.fit(x=train_x,
			  y=train_y,
			  batch_size=512,
			  validation_data=(valid_x, valid_y),
			  epochs=40,
			  callbacks=[stop_early, logPredictPer100epoch()]
			  )
			  
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	example_batch = test_x
	example_result = model.predict(example_batch)
	
	return (model, history, loss, val_loss, example_batch, example_result, y_pre_batch)


seed = 946063




model, history, loss, val_loss, example_batch, example_result, y_pre_batch = model_with_seed(seed)






res_dir = "res/individual/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
np.savetxt(res_dir + "01.y_pre_history.csv", y_pre_batch, delimiter=",")
pd.DataFrame({"loss": loss, "val_loss": val_loss}).to_csv(res_dir + "02.loss_history.csv")
pd.DataFrame({"real_y":test_y, "pred_y": tf.reshape(example_result, [-1])}).to_csv(res_dir + "03.res.csv")



plt.scatter(test_y,tf.reshape(example_result, [-1]))
plt.savefig("123.pdf")
plt.savefig("123.tiff")


y_pre = pd.Series(tf.reshape(example_result, [-1]))
y = pd.Series(test_y)

print(y.corr(y_pre))
print(spearmanr(y_pre, y))
print(pearsonr(y_pre, y))
