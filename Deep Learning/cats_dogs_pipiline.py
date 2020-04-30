# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:19:11 2020

@author: EmrahSariboz
"""


import pathlib
import os
import tensorflow as tf

img_dir = pathlib.Path('cats_and_dogs')

file_list = sorted([str(path) for path in img_dir.glob('*.jpg')])

print(file_list)

labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]


#Create dataset

ds_file_labels = tf.data.Dataset.from_tensor_slices((file_list, labels))

for item in ds_file_labels:
    print(item[0].numpy(), item[0].numpy().decode(), item[1].numpy())
    
    
def load_and_preprocess(path, labels):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0
    return image,labels

img_width, img_height = 120, 80

ds_images_labels = ds_file_labels.map(load_and_preprocess)