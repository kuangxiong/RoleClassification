# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   test_axis.py
@Time    :   2021/09/24 15:23:29
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import tensorflow as tf 
import numpy as np 

# data1 = tf.constant(np.arange(24).reshape(2,3,4), dtype='float64')
# print(data1)
# attention_mask = tf.constant([[1, 1 ,0], [1, 0, 0]], dtype='float64')
# mat = tf.linalg.diag(attention_mask)
# print(mat)
# print(data1.shape, mat.shape)
# mat1 = tf.matmul(mat, data1)
# print(mat1)
# data_tf = tf.reduce_sum(attention_mask, axis=1)
# # data_tf = tf.reduce_sum(mat1, axis=1)
# print(data_tf)

data1 = tf.constant(np.arange(2, 10).reshape(2, 4), dtype='float32')
print(data1)

data2 = tf.constant([[1], [2]], dtype='float32')
print(data2)

print(data1/data2)