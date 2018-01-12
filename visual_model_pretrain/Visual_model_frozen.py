from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.python.framework import graph_util


saver = tf.train.import_meta_graph('Visual_model.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('./'))


#output_node_names="dropout_2/Identity" #tensorboard's name
output_node_names="fully_layer_1/Relu" #tensorboard's name
output_graph_def = graph_util.convert_variables_to_constants(
	sess, # The session
	input_graph_def, # input_graph_def is useful for retrieving the nodes 
	output_node_names.split(",")  
)

output_graph="Visual_model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
	f.write(output_graph_def.SerializeToString())
 
sess.close()
