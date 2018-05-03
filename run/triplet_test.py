import os
import sys
import cPickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'util')
sys.path.insert(0,'model')
#add the file path to local path

import triplet

import loader
import data_manager
import xml_parser
import sklearn.metrics
import skimage.transform

if len(sys.argv)!=2:
    print('Usage: python triplet_test.py <config>')
    exit(0)

hyper_params=xml_parser.parse(sys.argv[1],flat=False)


# Construct the loader
loader_params=hyper_params['loader']
my_loader=loader.loader(path=loader_params['path'],train=loader_params['train_data'],test=loader_params['test_data'])

# Construct the data manager
data_manager_params=hyper_params['data_manager']
method = data_manager_params['sampling_method']

data_manager_params['loader']=my_loader
my_data_manager=data_manager.data_manager(data_manager_params)
# Test
test_params=hyper_params['test']
batches=test_params['batches']
model2load=test_params['model2load']

# Construct the network
network_params=hyper_params['network']
my_network=triplet.triplet_net(network_params)



test_case_num=0
test_right_num=0

def Resize(img):

    lenth = len(img)

    imag_list = []
    for i in range(lenth):
        image = np.reshape(img[i,:,:,:],[32,32,3])
        resized_img = skimage.transform.resize(image, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
        imag_list.append(resized_img)
    imag_list = np.reshape(imag_list,[-1,224,224,3])
    return imag_list

my_network.train_validate_test_init()
my_network.load_params(model2load)
Total_pre = []
Total_label = []
NMI_list=[]
Batch_list=[]
for batch_idx in xrange(batches):
    data_r,data_1,data_2,data_label=my_data_manager.get_triplet_ranked_instance('test',my_network.batch_size,method,dual=False)
    data_r = Resize(data_r)
    data_1 = Resize(data_1)
    data_2 = Resize(data_2)
    prediction,=my_network.test(data_r,data_1,data_2)
    prediction=np.argmax(prediction,axis=1)
    hit_bits=map(lambda x: 1 if x[0]==x[1] else 0, zip(data_label,prediction))
    Total_label = np.append(Total_label,data_label)
    Total_pre = np.append(Total_pre,prediction)
    test_case_num+=my_network.batch_size
    test_right_num+=np.sum(hit_bits)
    NMI = sklearn.metrics.normalized_mutual_info_score(Total_label, Total_pre)
    if(batch_idx%20==19):
        NMI_list.append(NMI)
        Batch_list.append(batch_idx)
        print('batch_idx = %d/%d,NMI=%.4F,hit =  %d/%d = %1.f%%, accumulated accuracy = %d/%d=%.1f%%\r'%(
        batch_idx+1,batches,NMI,np.sum(hit_bits),my_network.batch_size,float(np.sum(hit_bits))/float(my_network.batch_size)*100,
        test_right_num,test_case_num,float(test_right_num)/float(test_case_num)*100))
plt.plot(Batch_list,NMI_list)
plt.title("NMI")
plt.show()