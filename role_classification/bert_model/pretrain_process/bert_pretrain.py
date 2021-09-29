# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   transformers_pretrain.py
@Time    :   2021/09/29 15:29:33
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer, \
     LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
import torch
import tqdm
import random
import numpy as np
import os

from role_classification.bert_model.model_config import ModelConfig 
from role_classification.load_data import load_init_data


train_dataset, test_dataset = load_init_data(ModelConfig)
text_list = []

## 加载训练集和测试集语料
for i in range(len(train_dataset)):
    print(train_dataset[i])


# config = BertConfig(

#model = BertModel.from_pretrained("model_source/checkpoint-70000")
model = BertForMaskedLM(config=config)

res = []
for i in range(len(train)):
    res.append(train.iloc[i]['text_a']+" "+train.iloc[i]['text_b'])

for i in range(len(test)):
    res.append(test.iloc[i]['text_a'] +" "+ test.iloc[i]['text_b'])

pd.Series(res).to_csv('sentence_concat.txt', header=False,index=0)

dataset = LineByLineTextDataset(
    # 'bert_vocab',
    tokenizer=tokenizer,
    file_path="./sentence_concat.txt",
    block_size=32,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
training_args = TrainingArguments(
    output_dir="./pretrain_model_0316",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=64,
    save_steps=5000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("bert_pretrain_model_0311")
