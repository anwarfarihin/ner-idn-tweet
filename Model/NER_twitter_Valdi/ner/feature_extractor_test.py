import os.path
import numpy as np
import json
from gensim.models import Word2Vec
import logging
from keras.utils.np_utils import to_categorical
from word_embedding import WordEmbedding

import configparser
Config = configparser.ConfigParser()
Config._interpolation = configparser.ExtendedInterpolation()
Config.read('config.ini')

scenario = Config.get('general', 'scenario')
project = Config.get(scenario, 'project')
'''Extracting Features'''

logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def to_int(sents, dictionary_src):
	int_sents = []
	for sent in sents:
		int_sents.append(dictionary_src[sent])
	return int_sents

def extractData(filename):
	x = []
	with open(filename) as f:
		data = json.load(f)
		for sentence in data:
			x.append(sentence["text"])
	return x

file1 = './test/' + project + '/data.json'
x_train = extractData(file1)

we = WordEmbedding(model_name=Config.get(scenario, 'we_model'), dirpath='word-embedding/model')
x_train_we = [we.transform(sentence) for sentence in x_train]
x_train_context = [we.context_window(sentence, nb_contexts=3) for sentence in x_train_we] # bikin context window di sini. w1 w2 w3 --> w1<start>w2  w2w1w3 w3w2<end>

label_dict = {"B-ORGANIZATION": 0, 
			"I-ORGANIZATION": 1, 
			"B-LOCATION": 2, 
			"I-LOCATION": 3, 
			"B-PERSON": 4,
			"I-PERSON": 5,
			"O": 6
			}


dict_x_train = {}
for i, sentence in enumerate(x_train_we):
	dict_x_train[i] = sentence

dict_x_context = {}
for i, sentence in enumerate(x_train_context):
	dict_x_context[i] = sentence

outfile_x_train = open('./test/' + project + '/x_train.json', 'w')
outfile_x_train.write(json.dumps(dict_x_train))
outfile_x_train.close()

outfile_x_context = open('./test/' + project + '/x_train_context.json', 'w')
outfile_x_context.write(json.dumps(dict_x_context))
outfile_x_context.close()

outfile_x_human = open('./test/' + project + '/x_human.json', 'w')
outfile_x_human.write(json.dumps(x_train))
outfile_x_human.close()