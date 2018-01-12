from gensim.models import word2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("wiki_texts.txt")
    model = word2vec.Word2Vec(sentences, size=250, sg=1)

    #保存模型，供日後使用
    model.save("med250.model.bin")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model.bin")

if __name__ == "__main__":
    main()

