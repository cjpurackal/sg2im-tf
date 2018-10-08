import tensorflow as tf
import config.Config as conf
from gcn.hash import Hash
import numpy as np

tf.enable_eager_execution()

objects = ["car","bike","cycle"]
relationships = ["left of", "left of"]
num_objs = len(objects)
num_rels = len(relationships)

cfg = conf.Config("dummy")

obj_embeddings = tf.get_variable("object_embeddings",[num_objs, cfg.obj_embedding_size])
rel_embeddigs = tf.get_variable("relationships_embeddings",[num_rels, cfg.rel_embedding_size])

#dummy test
num_edges = 4
x = np.random.rand(num_edges,3*cfg.obj_embedding_size)
edges = np.array([[0,1],[1,2],[2,3],[3,0]])
s_idx= edges[:,0]
o_idx = edges[:,1]
# print (s_idx)


g = Hash("g", [3*cfg.obj_embedding_size, 100, cfg.gs_size+cfg.gp_size+cfg.go_size], "relu") # computes gs,gp,go
new_s_emb, vr_, new_o_emb = g.infer(x) #returns new embeddings for predicates and cadidate object vectors 
# print (s_idx.shape)
# print (new_s_emb.shape)
num_objects = max(s_idx)+1 if max(s_idx) > max(o_idx) else max(o_idx)+1

#pooling the candidate vectors
pool = tf.Variable(np.zeros([num_objects,new_s_emb.shape[1]]))
V_i_s = tf.scatter_add(pool,s_idx, new_s_emb)
V_i_o = tf.scatter_add(pool, o_idx, new_o_emb)
V = V_i_s + V_i_o

h = Hash("h", [V.shape[1], 100, cfg.new_obj_emb_size], "relu")
vi_ = h.infer(V) #returns new object embeddings
