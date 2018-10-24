# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-10-20 10:50:17
# @Last Modified by:   vamshi
# @Last Modified time: 2018-10-20 11:35:50

#This file creates dataset with stratified split and saves in ./datasets


import tensorflow as tf
import sklearn 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import scipy.io
import scipy
import numpy as np
import sys,os
import pickle


#create stratified mnist dataset
mnist = tf.keras.datasets.mnist.load_data()
X =np.concatenate((mnist[0][0],mnist[1][0]),axis=0)
y = np.concatenate((mnist[0][1],mnist[1][1]),axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
mnist_dict = {"X_train": X_train, "y_train":y_train, "X_train":X_train, "y_train":y_train, "X_val":X_val, "y_val":y_val}
if not os.path.exists("./datasets"):
	os.makedirs("./datasets")
with open("./datasets/mnist.pickle","wb") as f:
	pickle.dump(mnist_dict, f)
print(X_train.shape, X_test.shape, X_val.shape)

# print "train"
# for i in range(10):
# 	print(float((y_train==i).sum())/X_train.shape[0])
# print "test"
# for i in range(10):
# 	print(float((y_test==i).sum())/X_test.shape[0])
# print "val"
# for i in range(10):
# 	print(float((y_val==i).sum())/X_val.shape[0])


#cifar dataset
cifar10   = tf.keras.datasets.cifar10.load_data()
X = np.concatenate((cifar10[0][0],cifar10[1][0]),axis=0)
y = np.concatenate((cifar10[0][1],cifar10[1][1]),axis=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
cifar_dict = {"X_train": X_train, "y_train":y_train, "X_train":X_train, "y_train":y_train, "X_val":X_val, "y_val":y_val}
with open("./datasets/cifar.pickle","wb") as f:
	pickle.dump(cifar_dict, f)
print(X_train.shape, X_test.shape, X_val.shape)
# print "train"
# for i in range(10):
# 	print(float((y_train==i).sum())/X_train.shape[0])
# print "test"
# for i in range(10):
# 	print(float((y_test==i).sum())/X_test.shape[0])
# print "val"
# for i in range(10):
# 	print(float((y_val==i).sum())/X_val.shape[0])


#svhn dataset
svhn      = np.concatenate((scipy.io.loadmat('./svhn_data/train_32x32.mat')['X'],scipy.io.loadmat('./svhn_data/test_32x32.mat')['X']),axis=3)
X = svhn.transpose((3,0,1,2))
y = np.concatenate((scipy.io.loadmat('./svhn_data/train_32x32.mat')['y'],scipy.io.loadmat('./svhn_data/test_32x32.mat')['y']),axis=0)-1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
svhn_dict = {"X_train": X_train, "y_train":y_train, "X_train":X_train, "y_train":y_train, "X_val":X_val, "y_val":y_val}
with open("./datasets/svhn.pickle","wb") as f:
	pickle.dump(svhn_dict, f)
print(X_train.shape, X_test.shape, X_val.shape)
# print "train"
# for i in range(10):
# 	print(float((y_train==i).sum())/X_train.shape[0])
# print "test"
# for i in range(10):
# 	print(float((y_test==i).sum())/X_test.shape[0])
# print "val"
# for i in range(10):
# 	print(float((y_val==i).sum())/X_val.shape[0])
