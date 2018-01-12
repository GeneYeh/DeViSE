from gensim.models import word2vec
from gensim import models
import logging
import random
import numpy as np
from sklearn import preprocessing
def load_word2vec():
	global model
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = models.Word2Vec.load('med250.model.bin')
	return model
def label(input_label):
	if input_label == 0:
		x = preprocessing.normalize(model.wv['airplane'].reshape(1,250),norm='l2')
		return x
	elif input_label == 1:
		x = preprocessing.normalize(model.wv['automobile'].reshape(1,250),norm='l2')
		return x
	elif input_label == 2:
		x = preprocessing.normalize(model.wv['bird'].reshape(1,250),norm='l2')
		return x
	elif input_label == 3:
		x = preprocessing.normalize(model.wv['cat'].reshape(1,250),norm='l2')
		return x
	elif input_label == 4:
		x = preprocessing.normalize(model.wv['deer'].reshape(1,250),norm='l2')
		return x
	elif input_label == 5:
		x = preprocessing.normalize(model.wv['dog'].reshape(1,250),norm='l2')
		return x
	elif input_label == 6:
		x = preprocessing.normalize(model.wv['frog'].reshape(1,250),norm='l2')
		return x
	elif input_label == 7:
		x = preprocessing.normalize(model.wv['horse'].reshape(1,250),norm='l2')
		return x
	elif input_label == 8:
		x = preprocessing.normalize(model.wv['ship'].reshape(1,250),norm='l2')
		return x
	elif input_label == 9:
		x = preprocessing.normalize(model.wv['truck'].reshape(1,250),norm='l2')
		return x

def hit_word(vector,index):
	# for i in range(10): 
	acc = 0
	for i in range(100):
		embeeding = model.similar_by_vector(vector[i],topn=1)
		print("Data%d : Label = %d"%(i,index[i]),end='')
		for item in embeeding:
			#print(item[0])
			if(item[0] == 'airplane' and index[i] ==0):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'automobile' and index[i] ==1):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'bird' and index[i] ==2):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'cat' and index[i] ==3):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'deer' and index[i] ==4):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'dog' and index[i] ==5):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'frog' and index[i] ==6):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'horse' and index[i] ==7):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'ship' and index[i] ==8):
				print("   Hit!",end='')
				acc=acc+1
			elif(item[0] == 'truck' and index[i] ==9):
				#print("Index %d hit"%index[i],end='')
				print("   Hit!",end='')
				acc=acc+1
		print(" ")
				
	print("done!")
	return acc

model = load_word2vec()
embedding_vector = label(0)
for i in range(1,10):
	embedding_vector = np.vstack((embedding_vector,label(i)))
