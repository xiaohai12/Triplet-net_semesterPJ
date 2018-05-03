import os
#import cv2
import random
import struct
import numpy as np
from array import array
from scipy.io import loadmat
from keras.datasets import cifar10

class loader(object):

        #ubyte is type for minist image.

    def __init__(self,path,train='train_32x32.mat',test='test_32x32.mat'):
        self.train_file=os.path.join(path, train)
        self.test_file=os.path.join(path, test)

        self.image_size=None

        self.train_image_list=[]
        self.train_label_list=[]
        self.validate_image_list=[]
        self.validate_label_list=[]
        self.test_image_list=[]
        self.test_label_list=[]

    def load_train_set(self):
        images,labels=self.load("train")
        Shape = images.shape
        self.train_image_list = np.reshape(images,(-1,Shape[3],Shape[1],Shape[2]))
        self.train_label_list=np.reshape(labels,(len(labels),))-1
        #assert(len(self.train_image_list)==len(self.train_label_list))
        print 'there are %d training instances'%len(self.train_label_list)

    def load_test_set(self):
        images,labels=self.load("test")
        Shape = images.shape
        self.test_image_list=np.reshape(images,(-1,Shape[3],Shape[1],Shape[2]))
        self.test_label_list=np.reshape(labels,(len(labels),))-1
        #assert(len(self.test_image_list)==len(self.test_label_list))
        print 'there are %d testing instances'%len(self.test_label_list)

    def reserve_validation_set(self,prop):
        data_num=len(self.train_image_list)
        permute_list=np.random.permutation(np.arange(data_num))
        validate_size=0
        if prop>=1:
            validate_size=int(prop)
        elif prop>0:
            validate_size=int(prop*data_num)
        else:
            raise ValueError('size of validation set should be positive, now prop = %s'%str(prop))

        if validate_size>=data_num:
            raise ValueError('validation set is too large: prop = %s, available data instances = %d'%(str(prop),data_num))
        elif validate_size<=0:
            raise ValueError('validation set is too small: prop = %s, available data instances = %d'%(str(prop),data_num))

        train_idx_list=permute_list[validate_size:]
        validate_idx_list=permute_list[:validate_size]
        self.validate_image_list=self.train_image_list[validate_idx_list]
        self.validate_label_list=self.train_label_list[validate_idx_list]
        self.train_image_list=self.train_image_list[train_idx_list]
        self.train_label_list=self.train_label_list[train_idx_list]


    def load(self,method):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if method=="train":
            images,labels = x_train,y_train
            self.image_size = [images.shape[1],images.shape[2]]
            images=np.array(images).astype(np.float32)/256.			# Normalization
            labels=np.array(labels).astype(np.int)
        elif method =="test":
            images, labels = x_test, y_test
            self.image_size = [images.shape[1], images.shape[2]]
            images = np.array(images).astype(np.float32) / 256.  # Normalization
            labels = np.array(labels).astype(np.int)

        return images,labels

if __name__=='__main__':
    manager=loader('./data/MNIST')
    manager.load_train_set()
    manager.load_test_set()

    print 'there are %d training images and %d testing images'%(len(manager.train_image_list),len(manager.test_image_list))
    
#    cv2.namedWindow('image')
#    while True:
#        idx=random.randint(0,len(manager.train_image_list)+len(manager.test_image_list)-1)
#        image_info=manager.train_image_list[idx] if idx<len(manager.train_image_list) else manager.test_image_list[idx-len(manager.train_image_list)]
#        label_info=manager.train_label_list[idx] if idx<len(manager.train_image_list) else manager.test_label_list[idx-len(manager.train_image_list)]
#
#        print 'Label = ', label_info
#        while True:
#            cv2.imshow('image',image_info/255.)
#            key=cv2.waitKey(1) & 0xFF
#
#            if key==ord('z'):
#                exit(0)
#            if key==ord('c'):
#                break
