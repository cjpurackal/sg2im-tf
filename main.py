import tensorflow as tf
import numpy as np
import config.Config as conf
from models.gcn import GCN
from models.mlp import MLP
from data import sgparser as sgp

tf.enable_eager_execution()

objects = sgp.get_objects_unique("dataset/test_sg.json")
relationships = sgp.get_relationships_unique("dataset/test_sg.json")
edges = sgp.get_edges("dataset/test_sg.json")
num_objs = len(objects)
num_rels = len(relationships)
cgcn = conf.Config().GCN()

obj_embeddings = tf.get_variable("object_embeddings",[num_objs, cgcn.Din])
rel_embeddigs = tf.get_variable("relationships_embeddings",[num_rels, cgcn.Din])

obj_embs = [obj_embeddings[i] for i in objects]
vi = np.vstack(obj_embs)
pred_embs = [rel_embeddigs[i] for i in relationships]
vr = np.vstack(pred_embs)

model = GCN(4, cgcn, "relu")
vi_, vr_ = model.infer(vi, vr, edges)

boxnet = MLP("boxnet",[vi_.shape[1],cgcn.hidden_dim,cgcn.boxnet_out], batch_norm=False)
cords = boxnet.infer(vi_)
x1,y1,x2,y2 = cords[:,0], cords[:,1], cords[:,2], cords[:,3]