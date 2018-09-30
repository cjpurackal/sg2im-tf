import tensorflow as tf
import config.Config as conf
from gcn.hash import Hash

objects = ["car","bike","cycle"]
relationships = ["left of", "left of"]
num_objs = len(objects)
num_rels = len(relationships)

cfg = conf.Config("test config",[64, 64])

obj_embeddings = tf.get_variable("object_embeddings",[num_objs, cfg.obj_embedding_size])
rel_embeddigs = tf.get_variable("relationships_embeddings",[num_rels, cfg.rel_embedding_size])

g = Hash("g", [10,15], "relu") # computes gs,gp,go
h = Hash("h", [10, 30, 30], "relu") # symmetric function for computing new object vectors
