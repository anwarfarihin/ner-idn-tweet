import sklearn
import numpy as np
import pandas as pd

import logging

import torch
cuda_available = torch.cuda.is_available()

from simpletransformers.ner import NERModel, NERArgs
from nervaluate import Evaluator

# ==================================================================================
model_name='bert-base-multilingual-cased'


train_df =pd.read_csv("..\..\..\Dataset\\simpletransformers\\train.csv")
eval_df =pd.read_csv("..\..\..\Dataset\\simpletransformers\\val.csv")
test_df =pd.read_csv("..\..\..\Dataset\\simpletransformers\\test.csv")

label_list = [
    "O", 
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-PROD", "I-PROD",
    "B-WA", "I-WA",
    "B-EV", "I-EV",
]

# Create a NERModel
model_args=NERArgs()
model_args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",

    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
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

    "do_lower_case": False, #True if using uncased models
}

model = NERModel('bert', 
                model_name,
                args=model_args,
                labels=label_list,
                use_cuda=cuda_available,
                )

model.train_model(train_df,  eval_data=eval_df)

result, model_outputs, y_pred = model.eval_model(test_df)

df_x_test=test_df.groupby('sentence_id')['words'].apply(list)
df_y_test=test_df.groupby('sentence_id')['labels'].apply(list)
x_test=list(df_x_test)
y_test=list(df_y_test)

# Evaluate models with partial match and exact match
evaluator = Evaluator(y_test, y_pred, tags=['LOC', 'PER', 'ORG', 'PROD', 'WA', 'EV'], loader="list")
results, results_per_tag = evaluator.evaluate()

# export evaluation as file
with open("./results_1_batch.txt", "w") as f:
    str_results = repr(results)
    f.write(str_results)
    
with open("./results_per_tag_1_batch.txt", "w") as f:
    str_results_per_tag = repr(results_per_tag)
    f.write(str_results_per_tag)