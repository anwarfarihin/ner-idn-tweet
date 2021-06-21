import os.path
import numpy as np
import json
from gensim.models import Word2Vec
import logging

'''Train a Bidirectional LSTM.'''

logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

class WordEmbedding(object):

	def __init__(self, model_name, dirpath):
		self.model_name = model_name
		self.dirpath = dirpath
		print (model_name)
		print (dirpath)
		print ("{0}/{1}".format(self.dirpath, self.model_name))
		self.model = Word2Vec.load("../{0}/{1}".format(self.dirpath, self.model_name))
		self.hidden_size = self.model.wv.syn0[0].shape[0]

	def get_word_vector(self, word):
		word = word.lower()
		#return self.model[word]
		if self.model and word in self.model.wv.vocab:
			return self.model[word].tolist()
			
		else:
			return np.zeros((self.hidden_size,)).tolist()

	def transform(self, sentence):
		"""encode raw tokenize sentence (list of words) into feature
	    Parameter
	    ---------
	    tokens : list of tokenize sentence, e.g : ["tolong", "beliin", "tiket", "pesawat", "dong", "ke", "bali"]
	    """
		return [self.get_word_vector(word) for word in sentence]

	def context_window(self, sequence, nb_contexts):
		assert (nb_contexts % 2) == 1
		assert nb_contexts >= 1
		if isinstance(sequence, np.ndarray):
			sequence = sequence.tolist()
		padded = nb_contexts // 2 * [self.hidden_size*[0.]] + sequence + nb_contexts // 2 * [self.hidden_size*[0.]]
		out = []
		window = nb_contexts // 2
		for i in range(len(sequence)):
			#main = padded[i + window]
			#main = main.tolist()
			left = [item for sublist in padded[i:(i + window)] for item in sublist]
			right = [item for sublist in padded[(i+ window + 1):(i + nb_contexts)] for item in sublist]
			main = left + right
			out.append(main)
		#out = [[item for sublist in padded[i:(i + nb_contexts)] for item in sublist] for i in range(len(sequence))]
		assert len(out) == len(sequence)
		return out