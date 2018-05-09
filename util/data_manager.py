import numpy as np
import random

class data_manager(object):

    def __init__(self, hyper_param):
        '''
        >>> Construction function
        >>> hyper_param:
            >>> loader: dataset loader
            >>> class_num: int, number of categories
        '''
        self.loader=hyper_param['loader']
        self.class_num=hyper_param['class_num']
        self.validate_size=hyper_param['validate_size']
        self.channel = hyper_param['channel']
        self.loader.load_train_set()
        self.loader.load_test_set()
        if self.validate_size>0:
            self.loader.reserve_validation_set(prop=self.validate_size)
        self.image_size=self.loader.image_size

        self.train_image_set=self.loader.train_image_list
        self.train_label_set=self.loader.train_label_list
        self.validate_image_set=self.loader.validate_image_list
        self.validate_label_set=self.loader.validate_label_list
        self.test_image_set=self.loader.test_image_list
        self.test_label_set=self.loader.test_label_list
        self.set_pt={'train':0, 'validate':0, 'test':0}

        # Random Shuffle
        #number_of_train_instance=len(self.train_image_set)
        #number_of_validate_instance=len(self.validate_image_set)
        #number_of_test_instance=len(self.test_image_set)
        #random_shuffle_train_idx=np.random.permutation(np.arange(number_of_train_instance))
        #random_shuffle_validate_idx=np.random.permutation(np.arange(number_of_validate_instance))
        #random_shuffle_test_idx=np.random.permutation(np.arange(number_of_test_instance))
        #self.train_image_set=np.array(self.train_image_set)[random_shuffle_train_idx]
        #self.validate_image_set=np.array(self.validate_image_set)[random_shuffle_validate_idx]
        #self.test_image_set=np.array(self.test_image_set)[random_shuffle_test_idx]

        self.train_data_by_label=[[] for idx in xrange(self.class_num)]
        self.test_data_by_label=[[] for idx in xrange(self.class_num)]
        self.validate_data_by_label=[[] for idx in xrange(self.class_num)]

        for idx,(train_image,train_label) in enumerate(zip(self.train_image_set,self.train_label_set)):
            if train_label>=self.class_num:
                raise ValueError('Label Error in training set, index=%d, label=%d, class_num=%d'%(idx,train_label,self.class_num))
            self.train_data_by_label[train_label].append(train_image)

        for idx,(validate_image,validate_label) in enumerate(zip(self.validate_image_set,self.validate_label_set)):
            if validate_label>=self.class_num:
                raise ValueError('Label Error in validation set, index=%d, label=%d, class_num=%d'%(idx,validate_label,self.class_num))
            self.validate_data_by_label[validate_label].append(validate_image)

        for idx,(test_image,test_label) in enumerate(zip(self.test_image_set,self.test_label_set)):
            if test_label>=self.class_num:
                raise ValueError('Label Error in test set, index=%d, label=%d, class_num=%d'%(idx,test_label,self.class_num))
            self.test_data_by_label[test_label].append(test_image)

    def get_triplet_ranked_instance(self,set_label,number,method,dual=True):
        '''
        >>> get triplet training instances
        >>> set_label: str, label of subsets
        >>> number: int, number of instances
        >>> dual: optional bool,
        >>> method:str, sampling policy method
        >>> output:
            image_r: np.array of shape [number, height, width]
            image_1: np.array of shape [number, height, width]
            image_2: np.array of shape [number, height, width]
            label: np.array of shape [number,]
        '''

        image_r=np.zeros([number,self.channel,self.image_size[0],self.image_size[1]],dtype=np.float32)
        image_1=np.zeros([number,self.channel,self.image_size[0],self.image_size[1]],dtype=np.float32)
        image_2=np.zeros([number,self.channel,self.image_size[0],self.image_size[1]],dtype=np.float32)
        label=np.zeros([number,],dtype=np.float32)

        list_by_label=[]
        if set_label.lower() in ['train','training']:
            list_by_label=self.train_data_by_label
        elif set_label.lower() in ['test','testing']:
            list_by_label=self.test_data_by_label
        elif set_label.lower() in ['validate','validation']:
            list_by_label=self.validate_data_by_label
        else:
            raise ValueError('Unrecognized set label: %s'%set_label)

        image_idx=0
        while image_idx<number:
            majority_category = random.randint(0, self.class_num - 1)
            minority_category = random.randint(0, self.class_num - 1)
            while majority_category == minority_category:
                minority_category = random.randint(0, self.class_num - 1)

            majority_list = list_by_label[majority_category]
            minority_list = list_by_label[minority_category]

            instance_r = majority_list[random.randint(0, len(majority_list) - 1)]
            instance_1 = majority_list[random.randint(0, len(majority_list) - 1)]
            instance_2 = minority_list[random.randint(0, len(minority_list) - 1)]

            label[image_idx] = random.randint(0, 1)
            image_r[image_idx] = instance_r
            image_1[image_idx] = instance_1 if label[image_idx] == 0 else instance_2
            image_2[image_idx] = instance_2 if label[image_idx] == 0 else instance_1

            image_idx += 1

            if dual == True and image_idx < number:
                label[image_idx] = 1 - label[image_idx - 1]
                image_r[image_idx] = instance_r
                image_1[image_idx] = image_2[image_idx - 1]
                image_2[image_idx] = image_1[image_idx - 1]
                image_idx += 1

        permutation_idx=np.random.permutation(np.arange(number))
        image_r=image_r[permutation_idx]
        image_1=image_1[permutation_idx]
        image_2=image_2[permutation_idx]
        label=label[permutation_idx]

        return image_r,image_1,image_2,label

    def get_single_instance(self,set_label,number):
        '''
        >>> get a batch of single images from a specific subset
        >>> set_label: str, label of subset
        >>> number: number of images extracted
        >>> output:
            image: np.array of shape [number, height, width]
            label: np.array of shape [number,]
            end_of_epoch: boolean, indication of the end of epoch
        '''
        image_set=[]
        label_set=[]
        if set_label in ['train','training']:
            set_label='train'
            image_set=self.train_image_set
            label_set=self.train_label_set
        elif set_label in ['validate','validation']:
            set_label='validate'
            image_set=self.validate_image_set
            label_set=self.validate_label_set
        elif set_label in ['test','testing']:
            set_label='test'
            image_set=self.test_image_set
            label_set=self.test_label_set
        else:
            raise ValueError('Unrecognized set label: %s'%set_label)

        assert(len(image_set)==len(label_set))
        number_of_instances=len(image_set)



        end_of_epoch=False
        image=np.zeros([number,self.channel,self.image_size[0],self.image_size[1]],dtype=np.float32)
        label=np.zeros([number,],dtype=np.int)
        if self.channel==1:
            for idx in xrange(number):
                image[idx] = image_set[self.set_pt[set_label]]
                label[idx] = label_set[self.set_pt[set_label]]
                self.set_pt[set_label] += 1
                if self.set_pt[set_label] == number_of_instances:
                    end_of_epoch = True
                    self.set_pt[set_label] = 0
                    random_shuffle_idx = np.random.permutation(np.arange(number_of_instances))
                    image_set = image_set[random_shuffle_idx]
                    label_set = label_set[random_shuffle_idx]

        elif self.channel==2:
            for idx in xrange(number):
                image[idx,0]=image_set[self.set_pt[set_label]]
                rd_idx = random.randint(0,image_set.shape[0]-1)
                image[idx,1] = image_set[rd_idx]
                label[idx]=label_set[self.set_pt[set_label]]*10+label_set[rd_idx]
                self.set_pt[set_label]+=1
                if self.set_pt[set_label]==number_of_instances:
                    end_of_epoch=True
                    self.set_pt[set_label]=0
                    random_shuffle_idx=np.random.permutation(np.arange(number_of_instances))
                    image_set=image_set[random_shuffle_idx]
                    label_set=label_set[random_shuffle_idx]

        elif self.channel==3:
            for idx in xrange(number):
                image[idx]=image_set[self.set_pt[set_label]]
                label[idx] = label_set[self.set_pt[set_label]]
                self.set_pt[set_label]+=1
                if self.set_pt[set_label]==number_of_instances:
                    end_of_epoch=True
                    self.set_pt[set_label]=0
                    random_shuffle_idx=np.random.permutation(np.arange(number_of_instances))
                    image_set=image_set[random_shuffle_idx]
                    label_set=label_set[random_shuffle_idx]
        return image,label,end_of_epoch



