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
Config.read('config.ini')
scenario = Config.get('general', 'scenario')
###SETTING####

##############


def create_model(label_dict, timesteps, features, lstm_depth=1, kernel_size=3, filters=128, strides=1):
    input_layer = Input(shape=(timesteps, features))
    #masked_layer = Masking(mask_value=0.)(input_layer)
    cnn_layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
                       activation='relu', strides=strides)(input_layer)
    forward_layer = LSTM(units=128, return_sequences=True)(cnn_layer)
    backward_layer = LSTM(units=128, return_sequences=True,
                          go_backwards=True)(cnn_layer)
    merged_layer = merge([forward_layer, backward_layer],
                         mode='concat', concat_axis=-1)
    dropout_layer = Dropout(0.2)(merged_layer)

    output_layer = TimeDistributed(
        Dense(units=len(label_dict), activation='softmax'))(dropout_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    print('compiling model..')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    print(model.summary())
    return model


def create_model_with_wordshape(label_dict, shape_dict, timesteps, features, lstm_depth=1, kernel_size=3, filters=128, strides=1):
    input_layer = Input(shape=(timesteps, features))
    shape_layer = Input(shape=(timesteps, ), dtype='int32')
    shape_embedded = Embedding(len(shape_dict), 32,
                               input_length=timesteps)(shape_layer)
    merged_input = merge([input_layer, shape_embedded], mode='concat')
    #masked_layer = Masking(mask_value=0.)(input_layer)
    cnn_layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
                       activation='relu', strides=strides)(merged_input)
    forward_layer = LSTM(units=128, return_sequences=True)(cnn_layer)
    backward_layer = LSTM(units=128, return_sequences=True,
                          go_backwards=True)(cnn_layer)
    merged_layer = merge([forward_layer, backward_layer],
                         mode='concat', concat_axis=-1)
    dropout_layer = Dropout(0.2)(merged_layer)

    output_layer = TimeDistributed(
        Dense(units=len(label_dict), activation='softmax'))(dropout_layer)

    model = Model(inputs=[input_layer, shape_layer], outputs=[output_layer])
    print('compiling model..')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    print(model.summary())
    return model


def train_and_evaluate_model(model, x_train_vector, y_train_vector, mask_train_vector, x_test_vector, y_test_vector, mask_test_vector, label_dict, x_ws_train_vector, x_ws_test_vector):

    #model.fit(x_train_vector, y_train_vector, epochs=50, batch_size=50, sample_weight=mask_train_vector)
    if Config.get(scenario, 'wordshape') == 'True':
        model.fit([x_train_vector, x_ws_train_vector], y_train_vector, validation_data=([x_test_vector, x_ws_test_vector], y_test_vector, mask_test_vector),
                  epochs=50, batch_size=50, sample_weight=mask_train_vector)
    else:
        model.fit(x_train_vector, y_train_vector, validation_data=(x_test_vector, y_test_vector, mask_test_vector),
                  epochs=50, batch_size=50, sample_weight=mask_train_vector)
    dict_size = len(label_dict)

    conf_matrix = [[0 for i in range(dict_size)] for j in range(dict_size)]

    y_predicted_masked = []
    y_expected_masked = []

    if Config.get(scenario, 'wordshape') == 'True':
        for x_sequence, x_ws_sequence, y_sequence, mask_sequence in zip(x_test_vector, x_ws_test_vector, y_test_vector, mask_test_vector):
            y_predicted_sequence = model.predict(
                [np.array([x_sequence]), np.array([x_ws_sequence])])[0]

            y_predicted_sequence_masked = []
            y_expected_sequence_masked = []

            for h in range(len(y_predicted_sequence)):
                if mask_sequence[h] == 1:
                    y_pred = np.argmax(y_predicted_sequence[h])
                    y_predicted_sequence_masked.append(y_pred)
                    y_ori = np.argmax(y_sequence[h])
                    y_expected_sequence_masked.append(y_ori)

            y_predicted_masked.append(y_predicted_sequence_masked)
            y_expected_masked.append(y_expected_sequence_masked)
    else:
        for x_sequence, y_sequence, mask_sequence in zip(x_test_vector, y_test_vector, mask_test_vector):
            y_predicted_sequence = model.predict(np.array([x_sequence]))[0]

            y_predicted_sequence_masked = []
            y_expected_sequence_masked = []

            for h in range(len(y_predicted_sequence)):
                if mask_sequence[h] == 1:
                    y_pred = np.argmax(y_predicted_sequence[h])
                    y_predicted_sequence_masked.append(y_pred)
                    y_ori = np.argmax(y_sequence[h])
                    y_expected_sequence_masked.append(y_ori)

            y_predicted_masked.append(y_predicted_sequence_masked)
            y_expected_masked.append(y_expected_sequence_masked)

    print("Organization Evaluation")
    precision_org, recall_org, f1_org = evaluate(
        y_predicted_masked, y_expected_masked, 0, 1)
    print("Precision: ", precision_org)
    print("Recall: ", recall_org)
    print("F1: ", f1_org)
    print("----")

    print("Location Evaluation")
    precision_loc, recall_loc, f1_loc = evaluate(
        y_predicted_masked, y_expected_masked, 2, 3)
    print("Precision: ", precision_loc)
    print("Recall: ", recall_loc)
    print("F1: ", f1_loc)
    print("----")

    print("Person Evaluation")
    precision_per, recall_per, f1_per = evaluate(
        y_predicted_masked, y_expected_masked, 4, 5)
    print("Precision: ", precision_per)
    print("Recall: ", recall_per)
    print("F1: ", f1_per)
    print("----")

    return (f1_org, f1_loc, f1_per)


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
# x_train_vector = [[[x,x,x],[x,x,x],[x,x,x]] , [[x,x,x],[x,x,x],[x,x,x]]]
# y_train_vector = [[[x,x,x],[x,x,x],[x,x,x]] , [[x,x,x],[x,x,x],[x,x,x]]]
# train_mask_vector = [[[x,x,x],[x,x,x],[x,x,x]] , [[x,x,x],[x,x,x],[x,x,x]]]

# Read feature word embedding with Word2Vec
x_train_in = open_data('./features/' + project + '/x_train.json')

# Read feature context of word embedding with Word2Vec
if Config.get(scenario, 'context') == 'True':
    x_train_context_in = open_data(
        './features/' + project + '/x_train_context.json')

# Read feature POS TAG
if Config.get(scenario, 'pos') == 'True':
    x_train_pos_in = open_data('./features/' + project + '/x_train_pos.json')

# Read feature word shape
if Config.get(scenario, 'wordshape') == 'True':
    x_train_ws_in = open_data('./features/' + project + '/x_train_ws.json')

# Read shape dict
if Config.get(scenario, 'wordshape') == 'True':
    shape_dict = open_data('./features/' + project +
                           '/x_train_shape_dict.json')

# Read y
y_train_in = open_data('./features/' + project + '/y_train.json')

max_length = 0
x_train = []
x_ws_train = []
y_train = []
list_mask = []

for i in x_train_in.keys():

    # Add x feature word embedding
    x = x_train_in[i]

    # Add x feature context
    if Config.get(scenario, 'context') == 'True':
        x = merge_list(x, x_train_context_in[i])

    # Add x feature POS Tag
    if Config.get(scenario, 'pos') == 'True':
        x = merge_list(x, x_train_pos_in[i])

    x_train.append(x)

    # Find max length (timesteps) of sentence
    if max_length < len(x_train_in[i]):
        max_length = len(x_train_in[i])

    # Add x word shape feature in index
    if Config.get(scenario, 'wordshape') == 'True':
        x_ws_train.append(x_train_ws_in[i])

    # Add y train
    y_train.append(y_train_in[i])

print('max length: ', max_length)

for i in x_train_in.keys():
    sentence_len = len(x_train_in[i])
    padding = max_length - sentence_len
    mask = [1]*sentence_len + [0]*padding
    list_mask.append(mask)

# Padding for make the vector in same shape
padded_x_vector = pad_sequences(
    x_train, padding='post', maxlen=max_length, dtype='float32')
padded_y_vector = pad_sequences(
    y_train, padding='post', maxlen=max_length, dtype='float32')
padded_x_ws_vector = pad_sequences(
    x_ws_train, padding='post', maxlen=max_length, dtype='int32')

x_vector = np.array(padded_x_vector)
y_vector = np.array(padded_y_vector)
x_ws = np.array(padded_x_ws_vector)
mask_vector = np.array(list_mask)

print('dimensi x ', x_vector.shape)
print('dimensi y ', y_vector.shape)
print('dimensi mask ', mask_vector.shape)

label_dict = {0: "B-ORGANIZATION",
              1: "I-ORGANIZATION",
              2: "B-LOCATION",
              3: "I-LOCATION",
              4: "B-PERSON",
              5: "I-PERSON",
              6: "O"
              }

timesteps = max_length
features = len(x_vector[0][0])
kernel_size = 3
filters = 128
strides = 1
lstm_depth = 1

if Config.get(scenario, 'wordshape') == 'True':
    model = create_model_with_wordshape(label_dict=label_dict, shape_dict=shape_dict, timesteps=timesteps, features=features, lstm_depth=lstm_depth,
                                        kernel_size=kernel_size, filters=filters, strides=strides)
    model.fit([x_vector, x_ws], y_vector, epochs=50,
              batch_size=50, sample_weight=mask_vector)
else:
    model = create_model(label_dict=label_dict, timesteps=timesteps, features=features, lstm_depth=lstm_depth,
                         kernel_size=kernel_size, filters=filters, strides=strides)
    model.fit(x_vector, y_vector, epochs=50,
              batch_size=50, sample_weight=mask_vector)

# Save model
model_json = model.to_json()
open("./model/" + model_name + '.json', 'w').write(model_json)
model.save_weights("./model/" + model_name + '.h5')
