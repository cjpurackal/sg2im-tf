class Config:
	class GCN:
		def __init__(self):
			self.Din = 128
			self.Dout = 128
			self.hidden_dim = 512
			self.gs_size = 50
			self.gp_size = 100
			self.go_size = 200
			self.new_obj_emb_size=100
			self.batch_size = 5
			self.boxnet_out = 4
	class Layout:
		def __init__(self):
			self.H = 128
			self.W = 128