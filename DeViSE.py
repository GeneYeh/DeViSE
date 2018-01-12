from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import math
from tensorflow.python.framework import graph_util
from word2vec import *

#import dataset using Cifar-10
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data1= unpickle('data_batch_1')

X_train1 = data1[b'data']
Y_train1 = data1[b'labels']

data2= unpickle('data_batch_2')
X_train2 = data2[b'data']
Y_train2 = data2[b'labels']

data3= unpickle('data_batch_3')
X_train3 = data3[b'data']
Y_train3 = data3[b'labels']

data4= unpickle('data_batch_4')
X_train4 = data4[b'data']
Y_train4 = data4[b'labels']

data5= unpickle('data_batch_5')
X_train5 = data5[b'data']
Y_train5 = data5[b'labels']

test_data= unpickle('test_batch')
X_test = test_data[b'data']
Y_test = test_data[b'labels']

X_train = np.append(X_train1,X_train2,axis=0)
X_train = np.append(X_train,X_train3,axis=0)
X_train = np.append(X_train,X_train4,axis=0)
X_train = np.append(X_train,X_train5,axis=0)
X_train = np.append(X_train,X_train1,axis=0)
X_train = np.append(X_train,X_train2,axis=0)
X_train = np.append(X_train,X_train3,axis=0)
X_train = np.append(X_train,X_train4,axis=0)
X_train = np.append(X_train,X_train5,axis=0)


Y_train = np.append(Y_train1,Y_train2,axis=0)
Y_train = np.append(Y_train,Y_train3,axis=0)
Y_train = np.append(Y_train,Y_train4,axis=0)
Y_train = np.append(Y_train,Y_train5,axis=0)
Y_train = np.append(Y_train,Y_train1,axis=0)
Y_train = np.append(Y_train,Y_train2,axis=0)
Y_train = np.append(Y_train,Y_train3,axis=0)
Y_train = np.append(Y_train,Y_train4,axis=0)
Y_train = np.append(Y_train,Y_train5,axis=0)

'''
Training data


'''
X = X_train.reshape( [-1, 3, 32 , 32])/255.0
X.astype(np.float32)
X = X.transpose([0,2,3,1])

#one-hot encoding
#Y divide into 10 categories.
Y = np.zeros((100000, 10))
Y[range(100000), Y_train] = 1

#add 50000 mirror images to dataset, so now dataset have 50000 original images and 50000 mirror images
for i in range(50000, 100000):
	X[i] = np.fliplr(X[i])


X_test_1 = X_test.reshape( [-1, 3, 32 , 32])/255.0
X_test_1.astype(np.float32)
X_test_1 = X_test_1.transpose([0,2,3,1])

Y_test_1 = np.zeros((10000, 10))
Y_test_1[range(10000), Y_test] = 1


label_index = np.argmax(Y_test_1, axis = 0)
true_class_old=[]
batch_size = 64
epochs = 10
total_acc = 0

#evaluate loss that DeViSE provided.
def get_loss(prediction, batch_size, ys):
	with tf.name_scope('evaluate_loss'):
		global true_class_old
		label_size = 10 
		logit_old = tf.matmul(embedding_vector ,tf.transpose(tf.nn.l2_normalize(prediction, dim=1)))
		logit = tf.transpose(logit_old)
    
		true_class_old = tf.argmax(ys, axis = 1)
		logit_flat = tf.reshape(logit, [-1])
		number = label_size * np.arange(batch_size, dtype=np.int64).reshape(-1)
		true_class = true_class_old + number
		t_label = tf.reshape(tf.gather(logit_flat, true_class), [1, batch_size])
		martix_t_label_one = tf.ones(shape = [label_size, 1], dtype = tf.float32)
		#
		martix_t_label = tf.matmul(martix_t_label_one, t_label)

		margin = 0.1
		margin_one = tf.ones(shape = [label_size, batch_size], dtype = tf.float32)
		#
		matrix_margin = margin * margin_one
		#
		matrix_logit = tf.transpose(logit)
		loss = (tf.reduce_sum(tf.nn.relu(matrix_margin - martix_t_label + matrix_logit)) - batch_size*margin) / batch_size
		return loss


frozen_graph="Visual_model.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
	restored_graph_def = tf.GraphDef()
	restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
	tf.import_graph_def(
		restored_graph_def,
		input_map=None,
		return_elements=None,
		name=""
	)

inputs = graph.get_tensor_by_name("fully_layer_1/Relu:0")
samples = graph.get_tensor_by_name("inputs/samples:0")

with tf.Session(graph=graph) as sess:
	ys = tf.placeholder(tf.float32, shape = [None, 10], name = "labels")
	
	prediction = tf.contrib.layers.fully_connected(
	inputs = inputs,
	num_outputs = 250,
	activation_fn = None,
	weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
	scope = "transformation"
	)

	loss = get_loss(prediction, batch_size, ys)
	Trainloss = tf.summary.scalar('Trainloss',loss)
    
	with tf.name_scope('train_step'):
		train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	
	sess.run(init)
    
	writer = tf.summary.FileWriter("logs/", sess.graph)
	
	for epoch in range(epochs):
		print("epoch:%d" %(epoch+1))
		i = 0
		j = batch_size
		for k in range(math.ceil(len(X)/batch_size) - 1):
			epoch_x, epoch_y= X[i:j], Y[i:j]
			_, loss_= sess.run([train_step, Trainloss], feed_dict = {samples: epoch_x, ys:epoch_y})
			i = i + batch_size
			j = j + batch_size
		writer.add_summary(loss_, epoch)
	a = 0
	b = 100
	acc = 0
	for l in range(math.ceil(len(X_test_1)/100)):
		print("Batch: ",l)
		epoch_x, epoch_y= X_test_1[a:b],Y_test_1[a:b]
		pred,label_index = sess.run([prediction, true_class_old], feed_dict = {samples: epoch_x, ys:epoch_y})
		a = a + 100
		b = b + 100
		accuracy = hit_word(pred,label_index)
		acc += accuracy
		print("Number of hit word: ",accuracy)	
	print(acc/10000)



