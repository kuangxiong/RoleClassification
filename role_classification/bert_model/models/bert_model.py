# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   Bert_model.py
@Time    :   2021/09/24 10:25:30
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''
import sys 
import tensorflow as tf 
import os 
from tensorflow import keras 
from transformers import TFBertModel, TFAutoModel

from role_classification.bert_model.model_config import ModelConfig

class InitBertConfig(ModelConfig):
    """
    原始bert模型超参数配置
    """
    dropout = 0.5
    n_class= 6

def bert_model(ModelConfig):
    """
    bert 模型网络骨架

    Args:
        ModelConfig ([class]): [Bert模型参数设置]
    """
    bert_model = TFBertModel.from_pretrained(ModelConfig.bert_path, from_pt=True)
    text_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='text_id')
    segment_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='segment')
    mask_id = tf.keras.layers.Input(shape=(ModelConfig.max_len,), dtype=tf.int32, name='mask')

    bert_output = bert_model([text_id, segment_id, mask_id], return_dict=True)
    first_pooler_output = bert_output['pooler_output']
    
    model_output = keras.layers.Dense(ModelConfig.n_class)(first_pooler_output)

    model = keras.Model(inputs=[text_id, segment_id, mask_id], outputs=[model_output])
    return model
    


