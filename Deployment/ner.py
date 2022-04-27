from simpletransformers.ner import NERModel, NERArgs
import sklearn
import numpy as np
import pandas as pd

import torch
cuda_available = torch.cuda.is_available()

label_list = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-PROD", "I-PROD",
    # "B-WA", "I-WA",
    # "B-EV", "I-EV",
]

model_path = 'bert-base-multilingual-cased'

# Create a NERModel
model_args = NERArgs()

model_args = {
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

    "do_lower_case": False,
}

model = NERModel('bert',
                 model_path,
                 args=model_args,
                 labels=label_list,
                 use_cuda=cuda_available,
                 )
