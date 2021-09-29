# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   data_postprocess.py
@Time    :   2021/09/28 14:03:14
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import numpy as np 

def data_postprocess(data):
    """
    竞赛数据后处理，将模型的输出转化为0～3之间的整型，用于基于回归分析模型

    Args:
        data ([list]): [回归模型的输出]

    Returns:
        [list]: [将输出的float型转化为int类型]
    """
    res = []
    for i in range(len(data)):
        tmp_i = 10000
        res_i = 0
        for j in range(0, 4):
            if tmp_i > abs(data[i]-j):
                tmp_i = abs(data[i]-j)
                res_i = j 
        res.append(res_i)
    return res

if __name__=='__main__':
    tmp_list = [-1, -0.3, 0.4, 0.5, 1.6, 1.8]
    res_list = data_postprocess(tmp_list)
    print(res_list)

            