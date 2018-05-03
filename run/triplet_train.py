# -*- coding: utf-8 -*-
import cPickle
import os
import sys
import skimage.transform
import numpy as np
import sklearn.metrics
if sys.version_info.major==2:
    input=raw_input
elif sys.version_info.major==3:
    xrange=range
    from builtins import input

sys.path.insert(0,'util')
sys.path.insert(0,'model')

import triplet

import loader
import data_manager
import xml_parser
import matplotlib.pyplot as plt



if len(sys.argv)!=2:
    print('Usage: python triplet_train.py <config>')
    exit(0)
#(x_train,y_train),(x_test,y_test) = cifar10.load_data()

hyper_params=xml_parser.parse(sys.argv[1],flat=False)
#print(sys.argv[1])
# Construct the loader
loader_params=hyper_params['loader']
my_loader=loader.loader(path=loader_params['path'],train=loader_params['train_data'],test=loader_params['test_data'])

# Construct the data manager
data_manager_params=hyper_params['data_manager']
data_manager_params['loader']=my_loader
method = data_manager_params['sampling_method']
my_data_manager=data_manager.data_manager(data_manager_params)

# Construct the network
network_params=hyper_params['network']
my_network=triplet.triplet_net(network_params)

# Train
train_params=hyper_params['train']
batches=train_params['batches']
check_err_frequency=train_params['check_err_frequency']
validate_frequency=train_params['validate_frequency']
validate_batches=train_params['validate_batches']
model_saved_folder=train_params['model_saved_folder']
begin_batch_idx=train_params['begin_batch_idx'] if 'begin_batch_idx' in train_params else 0
model2load=train_params['model2load'] if 'model2load' in train_params else None
overwrite=train_params['overwrite'] if 'overwrite' in train_params else False

training_loss=[]
loss_information={'train':{},'validate':{}}
validate_min_error=1e8
validate_best_pt=-1
batch_size=my_network.batch_size

if os.path.exists(model_saved_folder+os.sep+'%s.pkl'%my_network.name):
    loss_information=cPickle.load(open(model_saved_folder+os.sep+'%s.pkl'%my_network.name))
    ckpt_list=loss_information['train'].keys()+loss_information['validate'].keys()
    latest_ckpt=np.max(ckpt_list)
    if latest_ckpt>begin_batch_idx:
        if overwrite==True:
            print('network of the same name is already stored and the information can NOT be used!')
            answer=input('overwrite? (y/N) >>> ')
            if not answer.lower() in ['yes','y']:
                print('Aborted!')
                exit(0)
            else:
                print('Overwritten')
                loss_information={'train':{},'validate':{}}
        else:
            raise Exception('network of the same name is already stored, to overwrite it, please set overwrite flag to be True')
    else:
        if overwrite==True:
            print('network of the same name is already stored but the information can be used!')
            answer=input('use it if you train the same network. (Y/n) >>> ')
            if answer.lower() in ['no','n']:
                loss_information={'train':{},'validate':{}}
        else:
            print('network of the same name is already stored but the information can be used!')
            print('automatically use the information saved!')


def Resize(img):

    lenth = len(img)

    imag_list = []
    for i in range(lenth):
        image = np.reshape(img[i,:,:,:],[32,32,3])
        resized_img = skimage.transform.resize(image, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
        imag_list.append(resized_img)
    imag_list = np.reshape(imag_list,[-1,224,224,3])
    return imag_list


if not os.path.exists(model_saved_folder):
    os.makedirs(model_saved_folder)

my_network.train_validate_test_init()
Total_label=[]
Total_pre=[]
Total_loss=[]
batch_list = []
train_plot_loss = []
count = 0
learning_rate = 0.0001
learning_rate_temp = 0.0001
if model2load!=None and os.path.exists(model2load):
    my_network.load_params(file2dump=model2load)
for batch_idx in xrange(begin_batch_idx,begin_batch_idx+batches):

    data_r,data_1,data_2,data_label=my_data_manager.get_triplet_ranked_instance('train',batch_size,method,dual=False)
    ##normalization:
    data_r = Resize(data_r)
    data_1 = Resize(data_1)
    data_2 = Resize(data_2)

    prediction, loss=my_network.train(data_r,data_1,data_2,data_label,learning_rate)
    reshape_prediction = prediction[:,0]
    training_loss.append(loss)

    Total_label = np.append(Total_label, data_label)
    Total_pre = np.append(Total_pre, reshape_prediction)
    sys.stdout.write('batch_idx=%d/%d, loss=%.4f \r'%(batch_idx+1,batches,loss))

    if (batch_idx+1)%check_err_frequency==0:

        print('batch=[%d,%d), average=%.4f'%(batch_idx+1-check_err_frequency, batch_idx+1, np.mean(training_loss[-check_err_frequency:])))
        batch_list.append(batch_idx)
        train_plot_loss.append(np.mean(training_loss[-check_err_frequency:]))
    if (batch_idx+1)%validate_frequency==0:
        my_network.dump_params(file2dump=model_saved_folder+os.sep+'%s_%d.ckpt'%(my_network.name,batch_idx+1))

        validate_loss=[]
        Total_label_val = []
        Total_pre_val = []
        print(learning_rate)

        for validate_batch_idx in xrange(validate_batches):

            data_r,data_1,data_2,data_label=my_data_manager.get_triplet_ranked_instance('validate',batch_size,method,dual=False)
            data_r = Resize(data_r)
            data_1 = Resize(data_1)
            data_2 = Resize(data_2)
            prediction,loss=my_network.validate(data_r,data_1,data_2,data_label,learning_rate)
            reshape_prediction = prediction[:, 0]
            Total_label_val = np.append(Total_label_val, data_label)
            Total_pre_val = np.append(Total_pre_val, reshape_prediction)
            NMI = sklearn.metrics.normalized_mutual_info_score(Total_label_val, Total_pre_val)
            validate_loss.append(loss)

            sys.stdout.write('Validation: batch_idx=%d/%d, loss=%.4f, avg_loss=%.4f \r'%(validate_batch_idx+1, validate_batches, loss, np.mean(validate_loss)))
        if np.mean(validate_loss)<validate_min_error:
            validate_min_error=np.mean(validate_loss)
            validate_best_pt=batch_idx+1
            count = 0
            #learning_rate = 0.001

        else:
            count +=1
            if count==3:
                learning_rate = learning_rate/2
                count = 0
                if learning_rate==learning_rate_temp/256:
                    learning_rate = learning_rate*256
                    learning_rate_temp = learning_rate

        print('')

print('Best validated checkpoint = %d'%validate_best_pt)
plt.plot(batch_list[4:],train_plot_loss[4:])

plt.title("learning curve")
plt.show()
my_network.train_validate_test_end()
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# import cv2
# data = loadmat("./data/SVHN/train_32x32.mat")
# data_train = data['X']
# image = data_train[:,:,:,2]
# print data['y'][2]
# plt.imshow(image)
# plt.show()