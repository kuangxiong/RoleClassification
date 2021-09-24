# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   model_train.py
@Time    :   2021/09/24 11:44:28
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import os 
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
from data_preprocess import BertDataPreload
from role_classification.bert_model.model_config import ModelConfig
from role_classification.bert_model.models.bert_model import InitBertConfig
from role_classification.bert_model.models.bert_model import bert_model

logger = ModelConfig.cust_logger

if __name__ =='__main__':
    bert_predata = BertDataPreload()
    train_data, test_data = bert_predata.load_data()
    logger.info("加载原始数据")

    data_X_ind, data_X_seg, data_X_mask, data_Y = bert_predata.bert_text2id(train_data)
    logger.info("原始数据转bert模型输入格式！")

    model = bert_model(InitBertConfig)
    adam = tf.keras.optimizers.Adam(ModelConfig.learning_rate)

    model.compile(
        loss = 'mse',
        optimizer = adam,
        metrics=['mse']
    )
    # 保存模型的目录
    save_path = ModelConfig.save_model
    # 若文件不存在，创建文件
    if os.path.exists(save_path):
        os.mkdir(save_path)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "bert_model",
        save_best_only=True,
        save_weights_only=True,
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=2,
        restore_best_weights=True
    )
    train_X_ind = tf.convert_to_tensor(data_X_ind)
    train_X_seg = tf.convert_to_tensor(data_X_seg)
    train_X_mask = tf.convert_to_tensor(data_X_mask)

    logger.info("开始模型训练！")
    # history = model.fit(
    #     [train_X_ind, train_X_mask, train_X_seg], 
    #     train_label, 
    #     epochs= 5,
    #     batch_size = ModelConfig.batch_size
    # )

    logger.info("完成模型训练！")
    model.save_weights("bert_model.h5")
    print(model.summary())
    print(data_X_ind[0])