import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from mpl_toolkits.mplot3d import axes3d, Axes3D

sys.path.insert(0,'./util')
sys.path.insert(0,'./model')

import triplet
import loader
import data_manager
import xml_parser

if len(sys.argv)!=2:
    print('Usage: python feature_extraction.py <config>')
    exit(0)

hyper_params=xml_parser.parse(sys.argv[1],flat=False)

# Construct the loader
loader_params=hyper_params['loader']
my_loader=loader.loader()
print('loader constructed')

# Construct the data manager
data_manager_params=hyper_params['data_manager']

data_manager_params['loader']=my_loader
my_data_manager=data_manager.data_manager(data_manager_params)
print('data_manager constructed')

# Construct the network
network_params=hyper_params['network']
my_network=triplet.triplet_net(network_params)
print('triplet network constructed')

# Feature extraction
feature_extraction_params=hyper_params['feature_extraction_params']
batches=feature_extraction_params['batches']
model2load=feature_extraction_params['model2load']

original_feature_matrix=[]
label_vector=[]

my_network.train_validate_test_init()
my_network.load_params(model2load)
for batch_idx in xrange(batches):
    image,label,end_of_epoch=my_data_manager.get_single_instance('test',my_network.batch_size)
    image_feature,=my_network.extract(image)
    original_feature_matrix+=list(image_feature)
    label_vector+=list(label)
    if end_of_epoch==True:
        break

num_of_images=len(label_vector)
print('image feature extraction completed, there are %d images in total'%(num_of_images))

inner_class_distances=[]
outter_class_distances=[]
for image_idx1 in xrange(num_of_images):
    for image_idx2 in xrange(image_idx1):
        distance=np.linalg.norm(np.array(original_feature_matrix[image_idx1])-np.array(original_feature_matrix[image_idx2]),2)
        if label_vector[image_idx1]==label_vector[image_idx2]:
            inner_class_distances.append(distance)
        else:
            outter_class_distances.append(distance)

print('Average distance within categories: %.2f'%np.mean(inner_class_distances))
print('Average distance across categories: %.2f'%np.mean(outter_class_distances))

original_feature_matrix=np.array(original_feature_matrix).astype(np.float32)
label_vector=np.array(label_vector).astype(np.int)
distance_matrix=euclidean_distances(original_feature_matrix)
distance_matrix=(distance_matrix+distance_matrix.T)/2.0             # To ensure that the matrix is symmetry

seed=np.random.RandomState(seed=3)
mds=manifold.MDS(n_components=2,max_iter=3000,eps=1e-9,random_state=seed,dissimilarity='precomputed',n_jobs=1)
projected_feature_matrix=mds.fit(distance_matrix).embedding_

class_num=np.max(label_vector)+1
projected_feature_by_label=[[] for idx in xrange(class_num)]
for feature_vector,label in zip(projected_feature_matrix,label_vector):
    projected_feature_by_label[label].append(feature_vector)

def get_color_list(number):
    colors=['b','g','r','c','m','y','k']
    if number<=7:
        return colors[:number]
    else:
        allowed_tokens=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
        for idx in xrange(number-7):
            parts=[allowed_tokens[np.random.randint(16)] for idx in xrange(6)]
            colors.append('#'+''.join(parts))
    return colors

# color_list=get_color_list(class_num)
# fig_handlers=[]
# fig_labels=[]
# fig = plt.figure()
# ax = Axes3D(fig)

# for idx,feature_matrix in enumerate(projected_feature_by_label):
#     feature_matrix=np.array(feature_matrix)
#     handler=ax.scatter(feature_matrix[:,0],feature_matrix[:,1],feature_matrix[:,2],color=color_list[idx])
#     fig_handlers.append(handler)
#     fig_labels.append(str(idx))
#
# plt.legend(fig_handlers,fig_labels,loc=0,scatterpoints=1)
# plt.show()

color_list=get_color_list(class_num)
fig_handlers=[]
fig_labels=[]
Label_list = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
for idx,feature_matrix in enumerate(projected_feature_by_label):
    feature_matrix=np.array(feature_matrix)
    if feature_matrix ==[]:
        continue
    handler=plt.scatter(feature_matrix[:,0],feature_matrix[:,1],color=color_list[idx])
    fig_handlers.append(handler)
    Label = Label_list[idx]
    fig_labels.append(Label)
plt.legend(fig_handlers,fig_labels,scatterpoints=1)
plt.title("Embedding picture")
plt.show()


