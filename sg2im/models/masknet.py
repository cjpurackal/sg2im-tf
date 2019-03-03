import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation


class MaskNet:

	def __init__(self):
		self.model = Sequential()
		self.num_of_blocks = 4
		for i in range(self.num_of_blocks):
			self.model.add(UpSampling2D(size=(2,2), data_format="channels_first"))
			self.model.add(BatchNormalization())
			self.model.add(Conv2D(2, (3,3), padding="same"))
			self.model.add(Activation("relu"))
		self.model.add(Conv2D(16, (128,1)))
		self.model.add(Activation("sigmoid"))

	def infer(self, obj_emb):
		x = tf.reshape(obj_emb, [obj_emb.shape[0],obj_emb.shape[1],1,1])
		y = self.model.predict(x,batch_size=2)
		return y


	def print(self):
		if self.model is None:
			print ("please initialize model first")
		else:
			print (self.model.summary())
