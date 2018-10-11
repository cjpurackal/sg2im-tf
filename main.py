import tensorflow as tf
import numpy as np
import config.Config as conf
from models.gcn import GCN
from data import sgparser as sgp

tf.enable_eager_execution()

objects = sgp.get_objects_unique("dataset/test_sg.json")
relationships = sgp.get_relationships_unique("dataset/test_sg.json")
edges = sgp.get_edges("dataset/test_sg.json")
num_objs = len(objects)
num_rels = len(relationships)
cfg = conf.Config("dummy")

obj_embeddings = tf.get_variable("object_embeddings",[num_objs, cfg.obj_embedding_size])
rel_embeddigs = tf.get_variable("relationships_embeddings",[num_rels, cfg.rel_embedding_size])

obj_embs = [obj_embeddings[i] for i in objects]
obj_embs = np.vstack(obj_embs)
pred_embs = [rel_embeddigs[i] for i in relationships]
pred_embs = np.vstack(pred_embs)
model = GCN(4, cfg, "relu")
model.infer(obj_embs, pred_embs, edges)

# boxnet = MLP("boxnet",[vi_.shape[1],100,cfg.boxnet_out])
# cords = boxnet.infer(vi_)	
# x,y,w,h = cords[:,0], cords[:,1], cords[:,2], cords[:,3]