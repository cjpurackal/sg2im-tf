import tensorflow as tf
import numpy as np
import config.Config as conf
from gcn.mlp import MLP
import data.SGParser as sgp


tf.enable_eager_execution()

objects = sgp.get_objects_unique("data/test_sg.json")
relationships = sgp.get_relationships_unique("data/test_sg.json")
num_objs = len(objects)
num_rels = len(relationships)
cfg = conf.Config("dummy")

#creating embeddings
obj_embeddings = tf.get_variable("object_embeddings",[num_objs, cfg.obj_embedding_size])
rel_embeddigs = tf.get_variable("relationships_embeddings",[num_rels, cfg.rel_embedding_size])

#test on rea data
edges = sgp.get_edges("data/test_sg.json")
num_edges = len(edges)
x = np.zeros((num_edges,3*cfg.obj_embedding_size))
for i,edge in enumerate(edges):
	x[i,0:cfg.obj_embedding_size]=obj_embeddings[edge[0]]
	x[i,cfg.obj_embedding_size:cfg.obj_embedding_size+cfg.rel_embedding_size]=rel_embeddigs[edge[1]]
	x[i,cfg.obj_embedding_size+cfg.rel_embedding_size:]=rel_embeddigs[edge[2]]


edges = np.array(edges)
s_idx= edges[:,0]
p_idx = edges[:,1]
o_idx = edges[:,2]

g = MLP("g", [3*cfg.obj_embedding_size, 100, cfg.gs_size+cfg.gp_size+cfg.go_size], "relu") # computes gs,gp,go
new_s_emb, vr_, new_o_emb = g.infer(x) #returns new embeddings for predicates and cadidate object vectors 
# print (s_idx.shape)
# print (new_s_emb.shape)
num_objects = max(s_idx)+1 if max(s_idx) > max(o_idx) else max(o_idx)+1

#pooling the candidate vectors
pool = tf.Variable(np.zeros([num_objects,new_s_emb.shape[1]]))
V_i_s = tf.scatter_add(pool,s_idx, new_s_emb)
V_i_o = tf.scatter_add(pool, o_idx, new_o_emb)
V = V_i_s + V_i_o

h = MLP("h", [V.shape[1], 100, cfg.new_obj_emb_size], "relu")
vi_ = h.infer(V) #returns new object embeddings

boxnet = MLP("boxnet",[vi_.shape[1],100,cfg.boxnet_out])
cords = boxnet.infer(vi_)
x,y,w,h = cords[:,0], cords[:,1], cords[:,2], cords[:,3]
