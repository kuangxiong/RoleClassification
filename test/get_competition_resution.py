# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   get_competition_resution.py
@Time    :   2021/09/28 14:27:24
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
import pandas as pd 
from tensorflow import keras
from transformers import BertModel, BertTokenizer
from role_classification.bert_model.data_postprocess import data_postprocess
from role_classification.bert_model.models.bert_cls_mse  import InitBertConfig, bert_model
from role_classification.load_data import load_init_data
from config import GlobalData

logger = InitBertConfig.cust_logger

if __name__=='__main__':
    # 测试用例
    start_time = time.time()
    
    train_data, test_data = load_init_data(GlobalData)
    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(InitBertConfig.bert_path)
    model = bert_model(InitBertConfig)
    ckpt = tf.train.Checkpoint(model=model)

    # .expect_partial()去掉不需要的信息，比如优化器参数等信息
    ckpt.restore(tf.train.latest_checkpoint(InitBertConfig.save_model)).expect_partial()
    print(model.summary())

    id_list, emotion_list = [], []
    for i in range(len(test_data)):
        id_list.append(test_data[i][0])
        try:
            tmp_str = "".join(test_data[i][1])
            token_output = tokenizer(
                    tmp_str, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=InitBertConfig.max_len
                )
            indices, segments = token_output['input_ids'], token_output['token_type_ids']
            mask_token = token_output['attention_mask']
            res = model.predict([np.array([indices]), np.array([segments]), np.array([mask_token])])
            res_post = data_postprocess(res[0])

            emotion_list.append(",".join(list(map(str, res_post))))
        except:
            emotion_list.append("0,0,0,0,0")
    print(id_list[:5])
    print(emotion_list[:5])
    ans = pd.DataFrame({"id":id_list, "emotion":emotion_list})
    ans.to_csv("test/data_out/ans_v1.txt", index=False, sep='\t')