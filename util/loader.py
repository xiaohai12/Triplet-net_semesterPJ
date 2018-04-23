import os
#import cv2
import random
import struct
import numpy as np
from array import array

class loader(object):

        #ubyte is type for minist image.

    def __init__(self,path,train_img='train-images.idx3-ubyte',train_lbl='train-labels.idx1-ubyte',
        test_img='t10k-images.idx3-ubyte',test_lbl='t10k-labels.idx1-ubyte'):
        self.train_img_file=os.path.join(path, train_img)
        self.train_lbl_file=os.path.join(path, train_lbl)
        self.test_img_file=os.path.join(path, test_img)
        self.test_lbl_file=os.path.join(path, test_lbl)

        self.image_size=None

        self.train_image_list=[]
        self.train_label_list=[]
        self.validate_image_list=[]
        self.validate_label_list=[]
        self.test_image_list=[]
        self.test_label_list=[]

    def load_train_set(self):
        images,labels=self.load(self.train_img_file,self.train_lbl_file)
        self.train_image_list=images
        self.train_label_list=labels
        assert(len(self.train_image_list)==len(self.train_label_list))
        print 'there are %d training instances'%len(self.train_label_list)

    def load_test_set(self):
        images,labels=self.load(self.test_img_file,self.test_lbl_file)
        self.test_image_list=images
        self.test_label_list=labels
        assert(len(self.test_image_list)==len(self.test_label_list))
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

    def load(self,img_file,lbl_file):
        with open(img_file,'rb') as fopen:
            magic,size,rows,columns=struct.unpack('>IIII',fopen.read(16))
            if magic!=2051:
                raise ValueError('Magic number mismatch, 2051 expected, %d got'%magic)
            image_data=array('B',fopen.read())

        with open(lbl_file,'rb') as fopen:
            magic,size=struct.unpack('>II',fopen.read(8))
            if magic!=2049:
                raise ValueError('Magic number mismatch, 2049 expected, %d got'%magic)
            labels=array('B',fopen.read())

        if self.image_size==None:
            self.image_size=[rows,columns]
        assert(self.image_size==[rows,columns])

        images=[]
        for idx in xrange(size):
            image_matrix=np.array(image_data[idx*columns*rows:(idx+1)*columns*rows]).reshape([rows,columns])
            images.append(image_matrix)

        images=np.array(images).astype(np.float32)/256.			# Normalization
        labels=np.array(labels).astype(np.int)

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
