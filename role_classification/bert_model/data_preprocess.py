# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   data_preprocess.py
@Time    :   2021/09/23 15:54:17
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''
import sys 
import os 
import numpy as np 
import tensorflow as tf 
from transformers import BertConfig, BertModel, BertTokenizer
from tensorflow import keras 

from role_classification.bert_model.model_config import ModelConfig
from role_classification.load_data import load_init_data


def label2id(label_list):
    """
    将类别数据转化为0，1编码，具体的将6类3个级别的向量转化为24维向量

    Args:
        label_list ([list]): [原始训练数据类别标签]

    Returns:
        [list]: [含0，1元素的18维向量类别标签，用于分类模型]
    """
    res = [0] * 24
    for i in range(len(label_list)):
        if label_list[i] > 0:
            res[4*i + label_list[i]] = 1
    return res

class BertDataPreload(ModelConfig):
    """
    bert模型数据预处理

    Args:
        ModelConfig ([type]): [description]
    """

    def load_data(self):
        """
        download training data and test data
        """
        train_data, test_data = load_init_data(ModelConfig)
        return train_data, test_data

    def bert_text2id(self, train_text, training=True, is_mse=True):
        """
        将字转化为字id

        Args:
            train_text ([list]): [文本数据]
            training (bool, optional): [数据是否用于训练]. Defaults to True.
        """
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # tokenizer = BertTokenizer(self.tokenizer_path)
        data_X_ind, data_X_seg, data_X_mask = [], [], []
        data_Y, test_id = [], []
        N = len(train_text)
        for i in range(N):
            if training == True:
                seg, label = train_text[i][0], train_text[i][1]
                # 用于回归模型标签
                if is_mse:
                    data_Y.append(list(map(int, str(label).split(','))))
                # 用于分类模型标签
                else:
                    data_Y.append(label2id(list(map(int, str(label).split(',')))))
            else:
                
                id, seg = train_text[i][0], train_text[i][1]
                test_id.append(id)
            token_output = tokenizer(
                    " ".join(seg), 
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.max_len
                )
            indices, segments = token_output['input_ids'], token_output['token_type_ids']
            mask_token = token_output['attention_mask']
            data_X_ind.append(indices)
            data_X_seg.append(segments)
            data_X_mask.append(mask_token)
        if training == True:
            return data_X_ind, data_X_seg, data_X_mask, data_Y 
        else:
            return data_X_ind, data_X_seg, data_X_mask, test_id

if __name__ == '__main__':
    BertData = BertDataPreload()
    train_data, test_data = BertData.load_data()
    print(train_data[0])
    data_X_ind, data_X_seg, data_X_mask, data_Y = BertData.bert_text2id(train_data, is_mse=False)
    print(data_X_ind[0])
    print(data_X_seg[0])
    print(data_X_mask[0])
    print(data_Y[:2])