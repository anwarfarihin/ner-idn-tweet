import json

import numpy as np
from keras.layers import Conv1D, Dense, Masking, LSTM, TimeDistributed, Activation, Bidirectional, Dropout, Input, merge, Embedding
from keras_contrib.layers import CRF
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from partial_evaluation import evaluate

import configparser
Config = configparser.ConfigParser()
Config._interpolation = configparser.ExtendedInterpolation()
Config.read('./config.ini')
print(Config.sections())
scenario = Config.get('general', 'scenario')
###SETTING####

##############


def open_data(path):
    data = {}
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def merge_list(list_a, list_b):
    merged = []
    for element_a, element_b in zip(list_a, list_b):
        merged.append(element_a + element_b)
    return merged


project = Config.get(scenario, 'project')
model_name = Config.get(scenario, 'model_name')

# loading the data

# Read real human data
x_human = open_data('./test/' + project + '/x_human.json')
# Read feature word embedding with Word2Vec
x_train_in = open_data('./test/' + project + '/x_train.json')

# Read feature context of word embedding with Word2Vec
if Config.get(scenario, 'context') == 'True':
    x_train_context_in = open_data(
        './test/' + project + '/x_train_context.json')

# Read feature POS TAG
if Config.get(scenario, 'pos') == 'True':
    x_train_pos_in = open_data('./test/' + project + '/x_train_pos.json')

# Read feature word shape
if Config.get(scenario, 'wordshape') == 'True':
    x_train_ws_in = open_data('./test/' + project + '/x_train_ws.json')


max_length = 0
x_train = []
x_ws_train = []

for i in range(len(x_train_in)):
    idx = str(i)
    # Add x feature word embedding
    x = x_train_in[idx]

    # Add x feature context
    if Config.get(scenario, 'context') == 'True':
        x = merge_list(x, x_train_context_in[idx])

    # Add x feature POS Tag
    if Config.get(scenario, 'pos') == 'True':
        x = merge_list(x, x_train_pos_in[idx])

    x_train.append(x)

    # Find max length (timesteps) of sentence
    if max_length < len(x_train_in[idx]):
        max_length = len(x_train_in[idx])

    # Add x word shape feature in index
    if Config.get(scenario, 'wordshape') == 'True':
        x_ws_train.append(x_train_ws_in[idx])

max_length = 37
print('max length: ', max_length)

# Padding for make the vector in same shape
padded_x_vector = pad_sequences(
    x_train, padding='post', maxlen=max_length, dtype='float32')
padded_x_ws_vector = pad_sequences(
    x_ws_train, padding='post', maxlen=max_length, dtype='int32')

x_vector = np.array(padded_x_vector)
x_ws = np.array(padded_x_ws_vector)

print('dimensi x ', x_vector.shape)

label_dict = {0: "B-ORGANIZATION",
              1: "I-ORGANIZATION",
              2: "B-LOCATION",
              3: "I-LOCATION",
              4: "B-PERSON",
              5: "I-PERSON",
              6: "O"
              }

# load model
json_file = open("./model/" + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./model/" + model_name + '.h5')
print("Model loaded")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'], sample_weight_mode='temporal')

result = []  # [{"text": [], "label": []},{"text":[], "label":[]}]

for i in range(len(x_human)):
    x_words = x_human[i]
    x_sequence = x_vector[i]

    if Config.get(scenario, 'wordshape') == 'True':
        x_ws_sequence = x_ws[i]
        y_predicted_sequence = model.predict(
            [np.array([x_sequence]), np.array([x_ws_sequence])])[0]
    else:
        y_predicted_sequence = model.predict(np.array([x_sequence]))[0]

    y_predicted_sequence_masked = []

    for h in range(len(x_words)):
        y_pred = np.argmax(y_predicted_sequence[h])
        prediction = label_dict[y_pred]
        y_predicted_sequence_masked.append(prediction)

    result.append({"text": x_words, "label": y_predicted_sequence_masked})

# dumps result to json
out = open('./output/' + model_name + '.json', 'w')
out.write(json.dumps(result))
out.close()
