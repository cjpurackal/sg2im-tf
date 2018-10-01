class Config:
	def __init__(self,name):
			self.name = name
			self.obj_embedding_size = 64
			self.rel_embedding_size = 64
			self.gs_size = 50
			self.gp_size = 100
			self.go_size = 200
			self.batch_size = 5