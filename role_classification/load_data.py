# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   load_data.py
@Time    :   2021/09/23 13:30:53
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import pandas as pd 
import numpy as np 
from config import GlobalData

logger = GlobalData.cust_logger


def str2list(input_str):
    """
    字符串转列表，将字符和数字整合在一起

    Args:
        input_str ([list]): [原始文本字符列表]

    Returns:
        [list]: [文本字符串转成列表后的结果]
    """
    res_list = []
    i, index = 0, 65
    
    dic = {input_str[0]:index}
    # res_list.append(chr(index))
    index += 1

    test_v1 = input_str[1]
    while(i < len(test_v1)):
        if i+1 < len(test_v1) and test_v1[i].isalpha() and test_v1[i+1].isdigit():
            newstr = test_v1[i] + test_v1[i+1]
            if newstr not in dic:
                dic[newstr] = index
                index += 1
            res_list.append(chr(dic[newstr]))
            i += 2
        if i < len(test_v1):
            res_list.append(test_v1[i])
        i += 1
    return res_list

def load_init_data(GlobalData):
    """
    加载原始数据

    Args:
        GlobalData ([class]): [原始数据的全局路径设置]
    """
    train_dataset, test_dataset = [], []
    train_data = pd.read_csv(GlobalData.train_file, sep='\t')
    test_data = pd.read_csv(GlobalData.test_file, sep='\t')
    
    train_data = train_data[(train_data['character'].notna()) & (train_data['emotions'].notna())]
    test_data = test_data[test_data['character'].notna()]

    N_train, N_test = len(train_data), len(test_data)

    for i in range(N_train):
        tmp_list =  str2list([train_data.iloc[i]['character']] + [train_data.iloc[i]['content']])

        train_dataset.append([tmp_list, train_data.iloc[i]['emotions']])
    
    for i in range(N_test):
        tmp_list =  str2list([test_data.iloc[i]['character']] + [test_data.iloc[i]['content']]) 

        test_dataset.append(tmp_list)

    logger.info("load data finish!")
    return train_dataset, test_dataset


if __name__=='__main__':
    train_dataset, test_dataset = load_init_data(GlobalData)
    print(train_dataset[10])