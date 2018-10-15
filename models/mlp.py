#hash function for the gcn
#computes gs,gp,go and h

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
import config.Config as conf

class MLP:

	def __init__(self, typ, layers, activation="relu", batch_norm = True, dropout=0):
		#constructs the mlp
		self.model = keras.Sequential()
		self.cgcn = conf.Config().GCN()
		self.type = typ

		for i in range(len(layers)):
			if i == 0:
				self.model.add(keras.layers.Dense(layers[i+1], input_shape=(layers[i],), activation=activation))
			elif i != len(layers)-1:
				self.model.add(keras.layers.Dense(layers[i+1]))
				if batch_norm:
					self.model.add(BatchNormalization())	
			if dropout>0:
				self.model.add(keras.layers.Dropout(dropout))

	def infer(self, x):
		y = self.model.predict(x, batch_size=5)
		if self.type == "g":
			gs = y[:,0:self.cgcn.hidden_dim]
			gp = y[:,self.cgcn.hidden_dim:self.cgcn.hidden_dim+self.cgcn.Dout]
			go = y[:,self.cgcn.hidden_dim+self.cgcn.Dout:]
			return gs, gp ,go
		elif self.type == "h" or self.type == "boxnet":
			return y


		

	def print(self):
		if self.model is None:
			print ("please initialize model first")
		else:
			print (self.model.summary())
