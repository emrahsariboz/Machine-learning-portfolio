# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:55:45 2020

@author: EmrahSariboz
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import tensorflow as tf

a = [1.1, 2.2, 7.7, 4.4, 5.5, 3.3]

ds = tf.data.Dataset.from_tensor_slices(a)


ds_batch = ds.batch(4)

for i, elem in enumerate(ds_batch, 1):
    print('Batch {} :'.format(i), elem.numpy())
    
    

#Combining tensors
    
tf.random.set_seed(1)

t_x = tf.random.uniform([4,3], dtype = tf.float32)
t_y = tf.range(4)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))


for i in ds_joint:
    print('x :', i[0].numpy() , 'y :', i[1].numpy())
    
    
#Apply user-defined function by calling the .map method

print('-'*50)


ds_tran = ds_joint.map(lambda x, y: (x*2 - 1.0, y))

for i in ds_joint:
    print('x :', i[0].numpy() , 'y :', i[1].numpy())
    
#Shuffle the dataset
    
ds_after_shuffle = ds_joint.shuffle(buffer_size = len(t_x))
print('-'*50)
for i in ds_after_shuffle:
    print('x :', i[0].numpy() , 'y :', i[1].numpy())
print('-'*50)
    
#Batch the original dataset   
ds = ds_joint.batch(batch_size = 3, drop_remainder = False)

batch_x, batch_y = next(iter(ds))

print('Batch-x- : \n', batch_x.numpy())
print('Batch-y-: ', batch_y.numpy())

#Batch the shuffled dataset

ds = ds_after_shuffle.batch(batch_size=3, drop_remainder = False)

batch_x, batch_y = next(iter(ds))

print('Batch-x-ds-after-shuffle : \n', batch_x.numpy())
print('Battch-y-ds-after-shuffle: ', batch_y.numpy())


print('-'*50)
#Batch and repeat
ds = ds_joint.batch(3).repeat(count = 2)

for i, (batch_x, batch_y) in enumerate(ds):
    print('Iteation ', i, '\nBatch_X shape: ', batch_x.shape, ' \n Batch_X :', batch_x.numpy(), '\n Batch_y ', batch_y.numpy())
    
#Repeat and batch
#Batch and repeat
ds = ds_joint.repeat(count = 2).batch(3)

for i, (batch_x, batch_y) in enumerate(ds):
    print('Iteation ', i, '\nBatch_X shape: ', batch_x.shape, ' \n Batch_X :', batch_x.numpy(), '\n Batch_y ', batch_y.numpy())
