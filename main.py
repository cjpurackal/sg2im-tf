import tensorflow as tf
import config.Config as conf
from gcn.hash import Hash
import numpy as np

objects = ["car","bike","cycle"]
relationships = ["left of", "left of"]
num_objs = len(objects)
num_rels = len(relationships)

cfg = conf.Config("dummy")

obj_embeddings = tf.get_variable("object_embeddings",[num_objs, cfg.obj_embedding_size])
rel_embeddigs = tf.get_variable("relationships_embeddings",[num_rels, cfg.rel_embedding_size])

#dummy test
x = np.random.rand(cfg.batch_size,3*cfg.obj_embedding_size)
g = Hash("g", [3*cfg.obj_embedding_size, 100, cfg.gs_size+cfg.gp_size+cfg.go_size], "relu") # computes gs,gp,go
gs, gp, go = g.infer(x)