#!/usr/bin/env python
# _*_ coding: utf-8
# author linbin date 2018-04-14
####把样本中的数据随机分成  训练集80% 测试机10% 验证集10%
####这个方法，随机生成的数字只是落在 0-99 之间的区间

import numpy as np

validation_percentage = 10
test_percentage = 10

###初始化各个数据集
validation_images = []; validation_labels = []
testing_images = []; testing_labels = []
training_images = []; training_labels = []

image_value= "XXX.jpg"
current_label = 1
chance = np.random.randint(100)    ####随机生成[0,99]之间的数
if chance < validation_percentage:  ####前百分之十的保存为验证集
    validation_images.append(image_value)
    validation_labels.append(current_label)
elif chance < (test_percentage + validation_percentage): ####把百分之十至百分之二十之间的保存为测试集
    testing_images.append(image_value)
    testing_labels.append(current_label)
else:                                       ######把后面百分之八十的保存为训练集。
    training_images.append(image_value)
    training_labels.append(current_label)

## 这个与上面的方法不同的是，是把索引的值随机重新排列，
#在10507个样本之间随机选取6600个，因为随机生成的值，会有重复的，所以上面的方法不适合，这个问题
r = np.random.permutation(10507)
image_0[r[:6600],:]


