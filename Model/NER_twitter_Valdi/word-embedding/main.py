# -*- coding: utf-8 -*-
# pylint:
# disable=no-name-in-module,W0201,W0403,W1201,C0411,C0111,C0325,C0301,C0103,C0112

import logging
from gensim.models import Word2Vec
import sys
import json
import re
import os

import configparser
Config = configparser.ConfigParser()
Config._interpolation = configparser.ExtendedInterpolation()
Config.read('config.ini')

we_model_name = int(Config.get('general', 'we_model_name'))
we_dimension = int(Config.get('general', 'dimension'))
we_window = int(Config.get('general', 'window'))
we_min_count = int(Config.get('general', 'min_count'))

logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield [x.lower().strip() for x in re.split(r'(\W+)?', line) if x.strip()]

sentences = MySentences('./data') # a memory-friendly iterator
#model = gensim.models.Word2Vec(sentences, size=200)

w2v = Word2Vec(sentences=sentences, size=we_dimension, window=we_window, min_count=we_min_count)
w2v.save('./model/' + we_model_name)