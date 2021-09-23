# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------------------
@File    :   config.py
@Time    :   2021/08/24 23:55:36
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
"""
import os
import sys
import time
from utils.log_conf import Logger

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CURT_T = time.strftime('%Y_%m_%d')

class GlobalData:
    """
    全局变量设置
    """
    train_file = os.path.join(BASE_PATH, "data_source/train_dataset_v2.tsv")
    test_file = os.path.join(BASE_PATH, "data_source/test_dataset.tsv")
    cust_logger = Logger(f"logs/test_{CURT_T}.log", level='INFO')
