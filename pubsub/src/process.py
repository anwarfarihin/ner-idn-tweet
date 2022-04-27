import logging
from matplotlib.pyplot import get
from simpletransformers.ner import NERModel, NERArgs
from nltk.tokenize import TweetTokenizer
from coder import JsonCoder
import json

import torch
cuda_available = torch.cuda.is_available()


class LogProcessor:

    def __init__(self, sink):
        self.sink = sink

    def process(self, message):
        self.sink.publish(message.data, message.attributes)
        return True


class NERProcessor:

    def __init__(self, sink):
        self.sink = sink  # misal load model
        self.model_name = 'Model\\simpletransformers\\7\\checkpoint-2000'

        self.tk = TweetTokenizer()

        self.model_args = NERArgs()
        self.model_args = {
            "output_dir": "outputs/",
            "cache_dir": "cache_dir/",

            "fp16": True,
            "fp16_opt_level": "O1",
            "max_seq_length": 512,
            "train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "eval_batch_size": 8,
            "num_train_epochs": 5,
            "weight_decay": 0,
            "learning_rate": 4e-5,

            "adam_epsilon": 1e-8,
            "warmup_ratio": 0.06,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,

            "logging_steps": 50,
            "save_steps": 2000,

            "overwrite_output_dir": True,
            "reprocess_input_data": False,
            "evaluate_during_training": True,

            "do_lower_case": False,  # True if using uncased models
        }

        self.label_list = [
            "O",
            "B-PER", "I-PER",
            "B-ORG", "I-ORG",
            "B-LOC", "I-LOC",
            "B-PROD", "I-PROD",
            "B-WA", "I-WA",
            "B-EV", "I-EV",
        ]
        self.model = NERModel('bert',
                              self.model_name,
                              args=self.model_args,
                              labels=self.label_list,
                              use_cuda=cuda_available,
                              )

    def process(self, message):
        # predictions= self.predict_batched(batch_size=8, data_size=1024)
        mess = json.loads(message.data.decode('utf-8'))
        predictions = self.predict(mess)
        message.data['named_entities'] = predictions
        self.sink.publish(message.data, message.attributes)
        return True

    def encode_entities(self, entities):
        list_of_entities = list()
        for ent in entities:
            temp_dict = dict()
            temp_dict['token'], temp_dict['entity'] = ent
            list_of_entities.append(temp_dict)
        return list_of_entities

    def get_entities(self, row):
        full_entities = []
        prev_tag = "O"  # init default tag
        for i in row:
            token_label = list(i.items())[0]
            start_tag = token_label[1][:1]
            tag = token_label[1][2:] if token_label[1] != 'O' else 'O'
            if tag == "O":
                prev_tag = tag
                continue

            if start_tag == 'B':  # Begin NE
                full_entities.append([token_label[0], tag])
            elif start_tag == 'I' and prev_tag == tag:  # Inside NE
                full_entities[-1][0] = full_entities[-1][0] + \
                    " " + token_label[0]
            prev_tag = tag
        return encode_entities(full_entities)

    def predict(self, data):
        # print(data)
        # print(data["text"])

        predictions, raw_outputs = self.model.predict(
            [self.tk.tokenize(data["text"])],
            split_on_space=False,
        )
        print(data["text"], predictions[0])
        return get_entities(predictions[0])

    # def compare_data(self, test_sentences, predictions):
    #     pred_labels = [[list(label_pred.values())[0] for label_pred in tweet_pred] for tweet_pred in predictions]
    #     true_labels = test_sentences[:1024].labels

    #     print(len(pred_labels), len(true_labels))
    #     diff_length=0
    #     for i in range(len(pred_labels)):
    #         if len(pred_labels[i])!=len(true_labels[i]):
    #             diff_length += 1
    #             print(test_sentences.loc[i][1])
    #             print(len(pred_labels[i]), len(true_labels[i]))
    #             print(f'pred_labels[{i}]     {pred_labels[i]}')
    #             print(f'true_labels[{i}]     {true_labels[i]}')
    #             print()
    #     print(diff_length)

    # def predict_batched(self, batch_size, data_size):
    #     batch_size=int(data_size/batch_size)
    #     predictions, raw_outputs= list(), list()
    #     for end in range(batch_size, data_size+batch_size, batch_size):
    #         start=end-batch_size
    #         temp_predictions, temp_raw_outputs = self.model.predict(
    #             test_sentences[start:end].tweet,
    #             split_on_space=False, # if the input are list of list
    #             )
    #         predictions.extend(temp_predictions)
    #         raw_outputs.extend(temp_raw_outputs)
    #     return predictions, raw_outputs
