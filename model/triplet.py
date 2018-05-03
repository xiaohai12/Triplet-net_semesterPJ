import math

import tensorflow as tf
import numpy as np
import sklearn;

class triplet_net(object):

    def __init__(self, hyper_params,restore_from=None):
        # '''
        # >>> construction function
        # '''
        self.name=hyper_params['name']
        ##self.pretrain_model = hyper_params['pretrain_model_load']
        self.height=hyper_params['height']
        self.width=hyper_params['width']
        self.channel = hyper_params['channel']
        self.batch_size=hyper_params['batch_size']
        filter_height=hyper_params['filter_height']
        filter_width=hyper_params['filter_width']
        pool_height=hyper_params['pool_height']
        pool_width=hyper_params['pool_width']
        self.pool_strides = hyper_params['pool_strides']
        filter_num=hyper_params['filter_num']
        self.conv_layer_num=np.min([len(filter_height),len(filter_width),len(filter_num),len(pool_height),len(pool_width)])
        self.filter_size=zip(filter_height[:self.conv_layer_num],filter_width[:self.conv_layer_num])
        self.pool_size=zip(pool_height[:self.conv_layer_num],pool_width[:self.conv_layer_num])
        self.filter_num=filter_num[:self.conv_layer_num]
        self.local_fc_neuron_num=hyper_params['local_fc_neuron_num']
        self.local_fc_layer_num=len(self.local_fc_neuron_num)
        self.feature_dimension=self.local_fc_neuron_num[-1]
        self.loss_type=hyper_params['loss_type']
        self.update_policy=hyper_params['update_policy']
        self.max_gradient_norm=1.0 if not 'max_gradient_norm' in hyper_params else hyper_params['max_gradient_norm']
        self.sess=None
        self.vgg16_npy_path = '/home/eric/EPFL/semester_project/tripletnet/pretrained_model/vgg16.npy'
        # Print the first part:
        layer_idx=1
        print('============Structure of Triplet Net============')
        print('Input size: (%d,%d)'%(self.height,self.width))
        for idx,((filter_height_idx,filter_width_idx),filter_num_idx,(pool_height_idx,pool_width_idx)) in enumerate(zip(self.filter_size,self.filter_num,self.pool_size)):
            print('Conv%d: filter=(%d,%d), #filters=%d'%(idx+1,filter_height_idx,filter_width_idx,filter_num_idx))
            print('Pool%d: kernel=(%d,%d)'%(idx+1,pool_height_idx,pool_width_idx))
        for idx,neuron_num in enumerate(self.local_fc_neuron_num):
            print('Local FC Layer%d: %d'%(idx+1,neuron_num))


        self.image_r=tf.placeholder(tf.float32,shape=[self.batch_size,self.height,self.width,self.channel])
        self.image_1=tf.placeholder(tf.float32,shape=[self.batch_size,self.height,self.width,self.channel])
        self.image_2=tf.placeholder(tf.float32,shape=[self.batch_size,self.height,self.width,self.channel])
        self.labels=tf.placeholder(tf.int32,shape=[self.batch_size,])

        self.learning_rate = tf.placeholder(tf.float32)

        self.feature_r,param_r=self.feature_extractor(self.image_r,reuse=False)
        self.feature_1,param_1=self.feature_extractor(self.image_1,reuse=True)
        self.feature_2,param_2=self.feature_extractor(self.image_2,reuse=True)

        if self.loss_type in ['mlp',]:
            self.global_fc_neuron_num=hyper_params['global_fc_neuron_num']
            self.global_fc_layer_num=len(self.global_fc_neuron_num) if self.global_fc_neuron_num!=None else 0
            for idx,neuron_num in enumerate(self.global_fc_neuron_num):
                print('Global FC Layer%d: %d'%(idx+1,neuron_num))

            # diff_1 = tf.abs(self.feature_1 - self.feature_r)
            # distance1 = tf.reduce_sum(tf.matmul(diff_1, tf.transpose(diff_1)), 1,keepdims=True)
            # diff_2 = self.feature_2 - self.feature_r
            # distance2 = tf.abs(tf.reduce_sum(tf.matmul(diff_2, tf.transpose(diff_2)), 1,keepdims=True))
            # global_feature_vector = tf.concat([diff_1,diff_2], axis=1)
            global_feature_vector=tf.concat([self.feature_r,self.feature_1,self.feature_2],axis=1)
            # concat the output of triplet network value
            projected_vector=self.mlp_classifier(global_feature_vector,reuse=False)
            with tf.variable_scope('decision'):
                input_dimension=self.global_fc_neuron_num[-1]
                output_dimension=2
                W=tf.get_variable(name='W',shape=[input_dimension,output_dimension],
                    initializer=tf.contrib.layers.xavier_initializer())


                    #initial the weight w and noise b
                b=tf.get_variable(name='b',shape=[output_dimension],
                    initializer=tf.truncated_normal_initializer(stddev=0.2))
                self.unnormalized_prediciton=tf.add(tf.matmul(projected_vector,W),b)
            #this is for loss function 1
            #self.labels = tf.placeholder(tf.int32, shape=[self.batch_size,2])
            #self.prediction = tf.nn.softmax(self.unnormalized_prediciton)
            #self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=self.unnormalized_prediciton, multi_class_labels=self.labels))

            self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unnormalized_prediciton, labels=self.labels))
            self.prediction=tf.nn.softmax(self.unnormalized_prediciton)




        else:
            raise ValueError('Unrecognized loss type: %s'%self.loss_type)





        # Optimizer
        if self.update_policy['name'].lower() in ['sgd', 'stochastic gradient descent']:
            #learning_rate=self.update_policy['learning_rate']
            learning_rate = self.learning_rate
            momentum=0.0 if not 'momentum' in self.update_policy else self.update_policy['momentum']
            self.optimizer=tf.train.MomentumOptimizer(learning_rate, momentum)
        elif self.update_policy['name'].lower() in ['adagrad',]:
            # learning_rate=self.update_policy['learning_rate']
            learning_rate = self.learning_rate
            initial_accumulator_value=0.1 if not 'initial_accumulator_value' in self.update_policy \
                else self.update_policy['initial_accumulator_value']
            self.optimizer=tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value)
        elif self.update_policy['name'].lower() in ['adadelta']:
            # learning_rate=self.update_policy['learning_rate']
            learning_rate = self.learning_rate
            rho=0.95 if not 'rho' in self.update_policy else self.update_policy['rho']
            epsilon=1e-8 if not 'epsilon' in self.update_policy else self.update_policy['epsilon']
            self.optimizer=tf.train.AdadeltaOptimizer(learning_rate, rho, epsilon)
        elif self.update_policy['name'].lower() in ['rms', 'rmsprop']:
            # learning_rate=self.update_policy['learning_rate']
            learning_rate = self.learning_rate
            decay=0.9 if not 'decay' in self.update_policy else self.update_policy['decay']
            momentum=0.0 if not 'momentum' in self.update_policy else self.update_policy['momentum']
            epsilon=1e-10 if not 'epsilon' in self.update_policy else self.update_policy['epsilon']
            self.optimizer=tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)
        elif self.update_policy['name'].lower() in ['adam']:
            # learning_rate=self.update_policy['learning_rate']
            learning_rate = self.learning_rate
            beta1=0.9 if not 'beta1' in self.update_policy else self.update_policy['beta1']
            beta2=0.999 if not 'beta2' in self.update_policy else self.update_policy['beta2']
            epsilon=1e-8 if not 'epsilon' in self.update_policy else self.update_policy['epsilon']
            self.optimizer=tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        else:
            raise ValueError('Unrecognized Optimizer Category: %s'%self.update_policy['name'])

        # Apply gradient clip
        print('Gradient clip is applied, max norm = %.2f'%self.max_gradient_norm)

        # gradients=self.optimizer.compute_gradients(self.loss)
        # clipped_gradients=[(tf.clip_by_value(grad,-self.max_gradient_norm,self.max_gradient_norm),var) for grad,var in gradients]
        var = param_r.append([W,b])
        self.update=self.optimizer.minimize(self.loss,var_list=var)

        print('Triplet Neural Network Constructed!!')




    def feature_extractor(self, input_data,reuse=False):
        # '''
        # >>> feature extractor
        # >>> input_data: tf.Variable of shape [self.batch_size, self.height, self.width, 1]
        # >>> return the feature vector
        # '''

        # with tf.variable_scope('feature_extractor', reuse=reuse):
        #
        #     input_feature_num=[self.channel,]+self.filter_num[:-1]
        #     output_feature_num=self.filter_num
        #     image_size=[self.height,self.width,self.channel]
        #     with tf.variable_scope('cnn',reuse=reuse):
        #         for conv_idx in xrange(self.conv_layer_num):
        #             W=tf.get_variable(name='W%d'%(conv_idx+1),shape=[self.filter_size[conv_idx][0],self.filter_size[conv_idx][1],input_feature_num[conv_idx],output_feature_num[conv_idx]],
        #                               initializer=tf.contrib.layers.xavier_initializer())
        #
        #             conv_output=tf.nn.conv2d(input_data,W,strides=[1,1,1,1],padding='SAME')
        #             norm = tf.layers.batch_normalization(conv_output)
        #             activation = tf.nn.relu(norm)
        #             pool_output = tf.nn.max_pool(activation,
        #                                          ksize=[1, self.pool_size[conv_idx][0], self.pool_size[conv_idx][1], 1],
        #                                          strides=[1, self.pool_strides[conv_idx], self.pool_strides[conv_idx],
        #                                                   1], padding='SAME')
        #
        #             input_data = pool_output
        #             image_size=[(image_size[dim_idx])/self.pool_size[conv_idx][dim_idx] for dim_idx in xrange(2)]
        #     image_vec_size=output_feature_num[-1]*np.prod(image_size)
        #     input_data=tf.reshape(input_data,shape=[self.batch_size,image_vec_size])
        #     input_neuron_num=[image_vec_size,]+self.local_fc_neuron_num[:-1]
        #     output_neuron_num=self.local_fc_neuron_num
        #     with tf.variable_scope('mlp',reuse=reuse):
        #         for mlp_idx in xrange(self.local_fc_layer_num):
        #             W=tf.get_variable(name='W%d'%(mlp_idx+1),shape=[input_neuron_num[mlp_idx],output_neuron_num[mlp_idx]],
        #                               initializer=tf.contrib.layers.xavier_initializer())
        #
        #             b=tf.get_variable(name='b%d'%(mlp_idx+1),shape=[output_neuron_num[mlp_idx],],
        #                 initializer=tf.truncated_normal_initializer(stddev=0.2))
        #             ##add batch_normalization
        #             x = tf.layers.batch_normalization(tf.add(tf.matmul(input_data,W),b))
        #             input_data=tf.layers.dropout(tf.nn.relu(x), rate=0.25)
        #             #input_data = tf.nn.relu(tf.add(tf.matmul(input_data, W), b))



        def max_pool(bottom, name):
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        def conv_layer(bottom, name,data_dict):
            with tf.variable_scope(name):  # CNN's filter is constant, NOT Variable that can be trained
                conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
                lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
                return lout


        data_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()

        vgg_mean = [103.939, 116.779, 123.68]

        # def __init__(self, vgg16_npy_path=None, restore_from=None):
        #     # pre-trained parameters
        #     self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        #     self.tfx = tf.placeholder(tf.float32, [20, 224, 224, 3])
        #     self.tfy = tf.placeholder(tf.int32, [20, ])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input_data* 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = conv_layer(bgr, "conv1_1",data_dict)
        conv1_2 = conv_layer(conv1_1, "conv1_2",data_dict)
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1",data_dict)
        conv2_2 = conv_layer(conv2_1, "conv2_2",data_dict)
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = conv_layer(pool2, "conv3_1",data_dict)
        conv3_2 = conv_layer(conv3_1, "conv3_2",data_dict)
        conv3_3 = conv_layer(conv3_2, "conv3_3",data_dict)
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = conv_layer(pool3, "conv4_1",data_dict)
        conv4_2 = conv_layer(conv4_1, "conv4_2",data_dict)
        conv4_3 = conv_layer(conv4_2, "conv4_3",data_dict)
        pool4 = max_pool(conv4_3, 'pool4')

        conv5_1 = conv_layer(pool4, "conv5_1",data_dict)
        conv5_2 = conv_layer(conv5_1, "conv5_2",data_dict)
        conv5_3 = conv_layer(conv5_2, "conv5_3",data_dict)
        pool5 = max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        with tf.variable_scope('mlp', reuse=reuse):
            self.flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
            W1 = tf.get_variable(name='W1', shape=[7 * 7 * 512, 256],initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name='b1', shape=[256, ], initializer=tf.truncated_normal_initializer(stddev=0.2))
            x = tf.layers.batch_normalization(tf.add(tf.matmul(self.flatten, W1), b1))
            W2 = tf.get_variable(name='W2', shape=[256, 64], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name='b2', shape=[64, ], initializer=tf.truncated_normal_initializer(stddev=0.2))
            input_data = tf.nn.relu(tf.add(tf.matmul(x, W2), b2))
            #self.prediction = tf.nn.softmax(self.out)
        return input_data,[W1,b1,W2,b2]


    def mlp_classifier(self, input_data, reuse=False):
        # '''
        # >>> mlp classifier
        # '''
        assert(self.loss_type=='mlp')
        with tf.variable_scope('mlp_classifier', reuse=reuse):
            input_neuron_num=[self.feature_dimension*3,]+self.global_fc_neuron_num[:-1]
            output_neuron_num=self.global_fc_neuron_num
            with tf.variable_scope('mlp',reuse=reuse):
                for mlp_idx in xrange(self.global_fc_layer_num):
                    W=tf.get_variable(name='W%d'%(mlp_idx+1),shape=[input_neuron_num[mlp_idx],output_neuron_num[mlp_idx]],
                        initializer=tf.contrib.layers.xavier_initializer())

                    b=tf.get_variable(name='b%d'%(mlp_idx+1),shape=[output_neuron_num[mlp_idx],],
                        initializer=tf.truncated_normal_initializer(stddev=0.2))
                    ##add batch_normalization
                    x = tf.layers.batch_normalization(tf.add(tf.matmul(input_data, W), b))
                    input_data =tf.layers.dropout(tf.nn.relu(x), rate=0.25)
                    #input_data=tf.nn.relu(tf.add(tf.matmul(input_data,W),b))
        return input_data

    def train_validate_test_init(self):
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def query_variable(self, name):
        with tf.variable_scope('', reuse=True):
            var=tf.get_variable(name)
            value,=self.sess.run([var])
            return value



    def train(self, data_r, data_1, data_2, data_label,learning_rate):
        data_r=np.array(data_r).reshape([self.batch_size,self.height,self.width,self.channel])
        data_1=np.array(data_1).reshape([self.batch_size,self.height,self.width,self.channel])
        data_2=np.array(data_2).reshape([self.batch_size,self.height,self.width,self.channel])

        data_label=np.array(data_label).reshape([self.batch_size,])

        train_dict={self.image_r:data_r, self.image_1:data_1, self.image_2:data_2, self.labels:data_label,self.learning_rate:learning_rate}
        self.sess.run([self.update],feed_dict=train_dict)
        prediction_this_batch,loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=train_dict)
        return prediction_this_batch, loss_this_batch

    def validate(self, data_r, data_1, data_2, data_label,learning_rate):

        data_r=np.array(data_r).reshape([self.batch_size,self.height,self.width,self.channel])
        data_1=np.array(data_1).reshape([self.batch_size,self.height,self.width,self.channel])
        data_2=np.array(data_2).reshape([self.batch_size,self.height,self.width,self.channel])
        data_label=np.array(data_label).reshape([self.batch_size,])
        #dddata = 1 - data_label
        #data_label = zip(data_label, dddata)

        validate_dict={self.image_r:data_r, self.image_1:data_1, self.image_2:data_2, self.labels:data_label,self.learning_rate:learning_rate}
        prediction_this_batch,loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=validate_dict)
        return prediction_this_batch, loss_this_batch

    def test(self, data_r, data_1, data_2):

        data_r=np.array(data_r).reshape([self.batch_size,self.height,self.width,self.channel])
        data_1=np.array(data_1).reshape([self.batch_size,self.height,self.width,self.channel])
        data_2=np.array(data_2).reshape([self.batch_size,self.height,self.width,self.channel])

        test_dict={self.image_r:data_r, self.image_1:data_1, self.image_2:data_2}
        prediction_this_batch,=self.sess.run([self.prediction,],feed_dict=test_dict)
        return prediction_this_batch,

    def extract(self,data):

        data=np.array(data).reshape([self.batch_size,self.height,self.width,self.channel])

        extract_dict={self.image_r:data}
        features,=self.sess.run([self.feature_r,],feed_dict=extract_dict)
        return features,

    def dump_params(self,file2dump):
        saver=tf.train.Saver()
        saved_path=saver.save(self.sess, file2dump)
        print('parameters are saved in file %s'%saved_path)

    def load_params(self,file2load):
        saver=tf.train.Saver()
        saver.restore(self.sess, file2load)

        print('parameters are imported from file %s'%file2load)

    def train_validate_test_end(self):
        self.sess.close()
        self.sess=None
