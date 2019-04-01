from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import json
import numpy as np


# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# 			['this', 'is', 'the', 'second', 'sentence'],
# 			['yet', 'another', 'sentence'],
# 			['one', 'more', 'sentence'],
# 			['and', 'the', 'final', 'sentence']]


# # fit a 2d PCA model to the vectors
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()

def create_model(X_text):

	# convert to the format acceptable to w2v
	X_text_new = []
	for text in X_text:
		content = text.split('\n')
		temp = []
		for con in content:
			temp.extend(con.split())
		X_text_new.append(temp)


	# train model
	model = Word2Vec(X_text_new, min_count=1)
	return X_text_new, model

def create_doc_vec(X_text, model):
	X = []
	# calculate mean of all words in the document to create a single vector for the document
	# can try min and max as well

	for document in X_text:
		if len(document) == 0:
			X.append(np.zeros(300))
			continue
		document_vec = np.array([list(model[word]) for word in document])
		document_vec_mean = np.mean(document_vec, axis=0)
		document_vec_max = np.max(document_vec, axis=0)
		document_vec_min = np.min(document_vec, axis=0)
		document_vec = np.concatenate((document_vec_mean, document_vec_max, document_vec_min))
		X.append(document_vec)

	X = np.array(X)
	return X


