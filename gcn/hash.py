#hash function for the gcn
#computes gs,gp,go and h

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization

class Hash:

	def __init__(self, type, layers, activation):
		#constructs the mlp
		self.model = keras.Sequential()
		for i in range(len(layers)):
			if i == 0:
				self.model.add(keras.layers.Dense(layers[i+1], input_shape=(layers[i],), activation=activation))
			elif i != len(layers)-1:
				self.model.add(keras.layers.Dense(layers[i+1]))
				self.model.add(BatchNormalization())

	def infer(self):
		print ("to do")
		

	def print(self):
		if self.model is None:
			print ("please initialize model first")
		else:
			print (self.model.summary())
