from gensim.models import word2vec
from gensim import models
import logging

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = models.Word2Vec.load('med250.model.bin')
	model.accuracy('questions-words.txt')

	

if __name__ == "__main__":
	main()
