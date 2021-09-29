# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   model_predict.py
@Time    :   2021/09/27 18:14:30
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import tensorflow as tf 
import numpy as np 
import time 
import os
import re 
from tensorflow import keras
from transformers import BertConfig, BertModel, BertTokenizer
from role_classification.bert_model.models.bert_cls_mse  import InitBertConfig, bert_model
from role_classification.bert_model.data_postprocess import data_postprocess

logger = InitBertConfig.cust_logger

if __name__=='__main__':
    # 测试用例
    test_case = [
        'A喜欢吃西瓜，但是B不喜欢吃',
        '如果爱有天意，A希望和B能够永远在一起'            
    ]

    start_time = time.time()

    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(InitBertConfig.bert_path)
    model = bert_model(InitBertConfig)
    ckpt = tf.train.Checkpoint(model=model)

    # .expect_partial()去掉不需要的信息，比如优化器参数等信息
    ckpt.restore(tf.train.latest_checkpoint(InitBertConfig.save_model)).expect_partial()
    print(model.summary())

    # 文字转数字
    for e in test_case:
        print(e)
        token_output = tokenizer(
                    e, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=InitBertConfig.max_len
                )
        # print(token_output)
        indices, segments = token_output['input_ids'], token_output['token_type_ids']
        mask_token = token_output['attention_mask']
        res = model.predict([np.array([indices]), np.array([segments]), np.array([mask_token])])
        print("-"*50)
        print(res)
        res_post = data_postprocess(res[0])
        print("后处理的结果:", res_post)
        print("-"*50)
        




 





