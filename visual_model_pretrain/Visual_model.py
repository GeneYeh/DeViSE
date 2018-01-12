from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
	tile = np.tile(weights, reps = (im.shape[0], im.shape[1], 1))
	return np.sum(tile * im, axis=2)

def crop_center(img, cropx, cropy):
	y,x,c = img.shape
	startx = x//2 - cropx//2
	starty = y//2 - cropy//2
	return img[starty:starty+cropy, startx:startx+cropx, :]

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

X = X_train.reshape( [-1, 3, 32 , 32])/255.0
X.astype(np.float32)
X = X.transpose([0,2,3,1])

Y = np.zeros((100000, 10))
Y[range(100000), Y_train] = 1

for i in range(50000, 100000):
	X[i] = np.fliplr(X[i])


#fliplr
#plt.subplot(2,1,1)
# plt.imshow(X[0])
# plt.subplot(2,1,2)
# plt.imshow(X[50000])
# plt.show()


# crop center
# a = crop_center(X[0], 20, 20)
# X[0] = np.pad(a, ((6,6),(6,6),(0,0)), mode = "constant", constant_values=0)
# print(X[0])
# print(X[0].shape)
# plt.subplot(2,1,1)
# plt.imshow(X[0])

# plt.show()


X_test_1 = X_test.reshape( [-1, 3, 32 , 32])/255
X_test_1.astype(np.float32)
X_test_1 = X_test_1.transpose([0,2,3,1])

Y_test_1 = np.zeros((10000, 10))
Y_test_1[range(10000), Y_test] = 1

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, shape = [None, 32, 32, 3], name = "samples")
	ys = tf.placeholder(tf.float32, shape = [None, 10], name = "labels")

prob1 = tf.placeholder_with_default(1.0, shape=(),name="prob1")
batch_size = 128
epochs = 1000

conv_1 = tf.layers.conv2d(
	inputs = xs,
	filters = 32,
	kernel_size = [3, 3],
	padding = "SAME",
	activation = tf.nn.relu,
	name = "conv_1"
	)

pool_1 = tf.layers.max_pooling2d(
	inputs = conv_1,
	pool_size = [2, 2],
	strides = 2, 
	padding = 'same', 
	name = "pool_1"
	)




conv_2 = tf.layers.conv2d(
	inputs = pool_1,
	filters = 32,
	kernel_size = [4,4],
	padding = "SAME",
	activation = tf.nn.relu,
	name = "conv_2"
	)



pool_2 = tf.layers.average_pooling2d(
	inputs = conv_2,
	pool_size = [2, 2],
	strides = 2, 
	padding = 'same', 
	name = "pool_2"
	)

conv_3 = tf.layers.conv2d(
	inputs = pool_2,
	filters = 64,
	kernel_size = [5,5],
	padding = "SAME",
	activation = tf.nn.relu,
	name = "conv_3"
	)

pool_3 = tf.layers.average_pooling2d(
	inputs = conv_3,
	pool_size = [2, 2],
	strides = 2, 
	padding = 'same', 
	name = "pool_3"
	)

normal = tf.contrib.layers.batch_norm(pool_3,decay = 0.95,activation_fn = tf.nn.relu)

flatten_1 = tf.contrib.layers.flatten(
	inputs = normal,
	scope = "flatten"
	)

fully_layer_1 = tf.contrib.layers.fully_connected(
	inputs = flatten_1,
	num_outputs = 512,
	activation_fn = tf.nn.relu,
	weights_initializer =  tf.contrib.layers.variance_scaling_initializer(),
	scope = "fully_layer_1"
	)


prediction = tf.contrib.layers.fully_connected(
	inputs = fully_layer_1,
	num_outputs = 10,
	activation_fn = tf.nn.softmax,
	weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
	scope = "softmax"
	)

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction))

with tf.name_scope('train_step'):
	train_step = tf.train.RMSPropOptimizer(0.0005).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(prediction, 1, name="Argmax_Pred"), tf.argmax(ys, 1, name="Y_Pred"), 
	name="Correct_Pred")
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name="Cast_Corr_Pred"), name="Accuracy")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

best_acc=0.80
with tf.Session() as sess:
	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(init)
	for epoch in range(epochs):
		print("epoch:%d" %epoch)
		i = 0
		j = batch_size
		for k in range(math.ceil(len(X)/batch_size)):
			epoch_x, epoch_y= X[i:j], Y[i:j]
			sess.run(train_step, feed_dict = {xs: epoch_x, ys:epoch_y,prob1:0.5})
			i = i + batch_size
			j = j + batch_size
		acc = sess.run(accuracy, feed_dict={xs: X_test_1, ys: Y_test_1})
		print(acc)
		# if acc > best_acc:
		# 	best_acc = acc                
		# 	save_path = saver.save(sess,"Visual_model.ckpt")
		# 	print("save file")





