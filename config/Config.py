class Config:
	def __init__(self,name, props):
		self.name = name
		self.obj_embedding_size = props[0]
		self.rel_embedding_size = props[1]