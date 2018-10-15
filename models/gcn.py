import tensorflow as tf
import numpy as np
from models.mlp import MLP


class GCN:
	
	def __init__(self, num_layers, cfg, mlp_activation):
		self.num_layers = num_layers
		self.mlp_activation = mlp_activation
		self.cfg = cfg

	def traverse(self, obj_emb, pred_embs, edges):
		num_edges = len(edges)
		obj_emb_size = obj_emb.shape[1]
		pred_emb_size = pred_embs.shape[1]

		x = np.zeros((num_edges,2*obj_emb_size+pred_emb_size))
		for i,edge in enumerate(edges):
			x[i,0:obj_emb_size]=obj_emb[edge[0]]
			x[i,obj_emb_size:obj_emb_size+pred_emb_size]=pred_embs[edge[1]]
			x[i,obj_emb_size+pred_emb_size:]=obj_emb[edge[2]]

		edges = np.array(edges)
		s_idx= edges[:,0]
		p_idx = edges[:,1]
		o_idx = edges[:,2]

		g = MLP("g", [3*self.cfg.Din, self.cfg.hidden_dim, 2*self.cfg.hidden_dim + self.cfg.Dout], self.mlp_activation) # computes gs,gp,go
		new_s_emb, vr_, new_o_emb = g.infer(x) #returns new embeddings for predicates and cadidate object vectors 
		num_objects = max(s_idx)+1 if max(s_idx) > max(o_idx) else max(o_idx)+1


		#pooling the candidate vectors
		pool = tf.Variable(np.zeros([num_objects,new_s_emb.shape[1]]))
		V_i_s = tf.scatter_add(pool,s_idx, new_s_emb)
		V_i_o = tf.scatter_add(pool, o_idx, new_o_emb)
		V = V_i_s + V_i_o
		h = MLP("h", [self.cfg.hidden_dim, self.cfg.hidden_dim, self.cfg.Dout], "relu")
		vi_ = h.infer(V) #returns new object embeddings

		return vi_, vr_


	def infer(self, obj_embs, pred_embs, edges):
		for _ in range(self.num_layers):
			obj_embs, pred_embs = self.traverse(obj_embs, pred_embs, edges)
			# print (obj_embs.shape)
			# print (pred_embs.shape)
		return obj_embs, pred_embs
		


