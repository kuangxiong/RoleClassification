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

from role_classification.bert_model.pretrain_process.pre_config import PreModelConfig
from role_classification.load_data import load_init_data

def create_pretrain_txtfile(ModelConfig):
    """
    创建预训练模型的语料文件

    Args:
        ModelConfig ([class]): [模型配置类，训练和测试文件路径]
    """

    train_dataset, test_dataset = load_init_data(ModelConfig)
    text_list = []

    ## 加载训练集和测试集语料
    for i in range(len(train_dataset)):
        text_list.append(" ".join(train_dataset[i][0]))

    for i in range(len(test_dataset)):
        text_list.append(" ".join(test_dataset[i][0]))

    text_file = os.path.join(ModelConfig.premodel_save_path, "train_test_text.txt")    
    pd.Series(text_list).to_csv(text_file, header=False, index=False)


def bert_model_pretrain(ModelConfig):
    """
    基于现有训练语料对原始bert模型进行预训练

    Args:
        ModelConfig ([class]): [原始bert模型配置类]
    """

    tokenizer = BertTokenizer.from_pretrained(ModelConfig.bert_path)
    model = BertModel.from_pretrained(ModelConfig.bert_path, from_pt=True)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(ModelConfig.train_test_dir, "train_test_text.txt"),
        block_size=32,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=ModelConfig.save_premodel,
        overwrite_output_dir=True,
        num_train_epochs=ModelConfig.n_epochs,
        per_device_train_batch_size=ModelConfig.batch_size,
        save_steps=ModelConfig.save_steps,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("bert_pretrain_model_0929")


if __name__=='__main__':
    create_pretrain_txtfile(PreModelConfig) 
    bert_model_pretrain(PreModelConfig)