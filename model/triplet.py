import tensorflow as tf
import numpy as np
import sklearn;

class triplet_net(object):

    def __init__(self, hyper_params):
        # '''
        # >>> construction function
        # '''
        self.name=hyper_params['name']
        self.height=hyper_params['height']
        self.width=hyper_params['width']
        self.channel = hyper_params['channel']
        self.batch_size=hyper_params['batch_size']
        filter_height=hyper_params['filter_height']
        filter_width=hyper_params['filter_width']
        pool_height=hyper_params['pool_height']
        pool_width=hyper_params['pool_width']
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

        self.feature_r=self.feature_extractor(self.image_r,reuse=False)
        self.feature_1=self.feature_extractor(self.image_1,reuse=True)
        self.feature_2=self.feature_extractor(self.image_2,reuse=True)



        # if self.loss_type in ['mlp',]:
        #     self.global_fc_neuron_num=hyper_params['global_fc_neuron_num']
        #     self.global_fc_layer_num=len(self.global_fc_neuron_num) if self.global_fc_neuron_num!=None else 0
        #     for idx,neuron_num in enumerate(self.global_fc_neuron_num):
        #         print('Global FC Layer%d: %d'%(idx+1,neuron_num))
        #
        #
        #     global_feature_vector=tf.concat([self.feature_r,self.feature_1,self.feature_2],axis=1)
        #     # concat the output of triplet network value
        #     projected_vector=self.mlp_classifier(global_feature_vector,reuse=False)
        #     with tf.variable_scope('decision'):
        #         input_dimension=self.global_fc_neuron_num[-1]
        #         output_dimension=2
        #         W=tf.get_variable(name='W',shape=[input_dimension,output_dimension],
        #                           initializer=tf.truncated_normal_initializer(stddev=0.2))
        #             #initial the weight w and noise b
        #         b=tf.get_variable(name='b',shape=[output_dimension],
        #             initializer=tf.truncated_normal_initializer(stddev=0.2))
        #         self.unnormalized_prediciton=tf.add(tf.matmul(projected_vector,W),b)
        #         print(self.unnormalized_prediciton)
        #
        #
        #     self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unnormalized_prediciton, labels=self.labels))
        #     self.prediction=tf.nn.softmax(self.unnormalized_prediciton)
        #
        #
        #
        # else:
        #     raise ValueError('Unrecognized loss type: %s'%self.loss_type)
        method = "Total"

        self.loss = self.f_divergence(method)






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
        gradients=self.optimizer.compute_gradients(self.loss)
        clipped_gradients=[(tf.clip_by_value(grad,-self.max_gradient_norm,self.max_gradient_norm),var) for grad,var in gradients]
        self.update=self.optimizer.apply_gradients(clipped_gradients)

        print('Triplet Neural Network Constructed!!')

    # def T_func(self,p,q,method):
    #     if method=="KL":
    #         return 1+tf.log(p/q)/np.log(2)
    #     elif method =="Total":
    #         return tf.sign(p/q-1)/2
    #     elif method =="squared":
    #         return 1-tf.sqrt(q/p)
    #     else:
    #         return tf.log(2*p/(p+q))/np.log(2)


    def conjugate(self,x,method):
        if method=="KL":
            return tf.exp(x - 1)

        elif method =="Total":
            return x

        elif method =="JS":
            return - tf.log(2 - tf.exp(x)) / np.log(2)
        else:
            return x / (1 - x)




    def f_divergence(self,method):
        inner_distance = []
        inter_distance = []
        Loss=0


        for i in range(self.batch_size):
            distance1 = self.feature_r[i, :]-self.feature_1[i, :]
            distance2 = self.feature_r[i, :]- self.feature_2[i, :]
            if self.labels[i] == 0:
                inner_distance.append(distance1)
                inter_distance.append(distance2)
            else:
                inner_distance.append(distance2)
                inter_distance.append(distance1)

        T_inner = tf.reduce_sum(self.mlp_classifier(inner_distance))
        T_inter = tf.reduce_sum(self.mlp_classifier(inter_distance,reuse=True))
        Loss =-T_inner+self.conjugate(T_inter,method)
        return Loss/self.batch_size






    def feature_extractor(self, input_data, reuse=False):
        # '''
        # >>> feature extractor
        # >>> input_data: tf.Variable of shape [self.batch_size, self.height, self.width, 1]
        # >>> return the feature vector
        # '''
        is_training = False
        with tf.variable_scope('feature_extractor', reuse=reuse):
            with tf.variable_scope('conv1') as scope:
                #W1 = tf.get_variable(name='Wconv1',shape=[5,5,3,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
                W = tf.get_variable(name='Wconv1', shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.2))
                b= tf.get_variable(name='bconv1', shape=[64],initializer=tf.truncated_normal_initializer(stddev=0.2))
                conv = tf.nn.conv2d(input_data, W, [1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, b)
                conv_1 = tf.nn.relu(pre_activation, name=scope.name)
                pool = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

                bn1 = tf.layers.batch_normalization(inputs=pool, center=True, scale=True, training=is_training)


            with tf.variable_scope('conv2') as scope:
                W = tf.get_variable(name='Wconv2', shape=[3,3,64,64],initializer=tf.truncated_normal_initializer(stddev=0.2))
                b = tf.get_variable(name='bconv2', shape=[64], initializer=tf.truncated_normal_initializer(stddev=0.2))
                conv = tf.nn.conv2d(bn1 , W, [1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, b)
                conv_2 = tf.nn.relu(pre_activation, name=scope.name)

                pool = tf.nn.max_pool(conv_2 , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')
                drop = tf.layers.dropout(inputs=pool, rate=0.25,training=is_training)

                bn2 = tf.layers.batch_normalization(inputs=drop , center=True, scale=True, training=is_training)


            with tf.variable_scope('conv3') as scope:

                W = tf.get_variable(name='Wconv3', shape=[3, 3, 64, 64],
                                     initializer=tf.truncated_normal_initializer(stddev=0.2))
                b = tf.get_variable(name='bconv3', shape=[64],
                                     initializer=tf.truncated_normal_initializer(stddev=0.2))
                conv = tf.nn.conv2d(bn2, W, [1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, b)
                conv_2 = tf.nn.relu(pre_activation, name=scope.name)

                pool = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name='pool2')
                drop = tf.layers.dropout(inputs=pool, rate=0.25, training=is_training)
                flat = tf.reshape(drop, [-1,64*4*4])
                bn2 = tf.layers.batch_normalization(inputs=flat, center=True, scale=True, training=is_training)

        return bn2



    def mlp_classifier(self, input_data, reuse=False):
        # '''
        # >>> mlp classifier
        # '''
        assert(self.loss_type=='mlp')
        with tf.variable_scope('mlp_classifier', reuse=reuse):

            with tf.variable_scope('local1',reuse=reuse) as scope:
                reshape = tf.reshape(input_data, [-1, 64*4*4])
                dim = reshape.get_shape()[1].value
                W = tf.get_variable(name="W",shape=[dim,64],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name='b', shape=[64], initializer=tf.truncated_normal_initializer(stddev=0.2))
                local3 = tf.nn.relu(tf.matmul(reshape, W) + b, name=scope.name)
                #drop = tf.nn.dropout(local3, 0.25)


            with tf.variable_scope('local2',reuse=reuse) as scope:
                W = tf.get_variable(name="W",shape=[64,32],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name='b', shape=[32], initializer=tf.truncated_normal_initializer(stddev=0.2))
                local4 = tf.nn.relu(tf.matmul(local3, W) + b, name=scope.name)


        return local4


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
        #dddata = 1-data_label
        #data_label = zip(data_label,dddata)
        data_label=np.array(data_label).reshape([self.batch_size,])

        # train_dict={self.image_r:data_r, self.image_1:data_1, self.image_2:data_2, self.labels:data_label,self.learning_rate:learning_rate}
        # self.sess.run([self.update],feed_dict=train_dict)
        # prediction_this_batch,loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=train_dict)
        # return prediction_this_batch, loss_this_batch

        train_dict = {self.image_r: data_r, self.image_1: data_1, self.image_2: data_2, self.labels: data_label,
                      self.learning_rate: learning_rate}
        self.sess.run([self.update], feed_dict=train_dict)
        loss_this_batch = self.sess.run([self.loss], feed_dict=train_dict)
        return loss_this_batch

    def validate(self, data_r, data_1, data_2, data_label,learning_rate):

        data_r=np.array(data_r).reshape([self.batch_size,self.height,self.width,self.channel])
        data_1=np.array(data_1).reshape([self.batch_size,self.height,self.width,self.channel])
        data_2=np.array(data_2).reshape([self.batch_size,self.height,self.width,self.channel])
        data_label=np.array(data_label).reshape([self.batch_size,])
        #dddata = 1 - data_label
        #data_label = zip(data_label, dddata)

        # validate_dict={self.image_r:data_r, self.image_1:data_1, self.image_2:data_2, self.labels:data_label,self.learning_rate:learning_rate}
        # prediction_this_batch,loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=validate_dict)
        # return prediction_this_batch, loss_this_batch

        validate_dict = {self.image_r: data_r, self.image_1: data_1, self.image_2: data_2, self.labels: data_label,
                         self.learning_rate: learning_rate}
        loss_this_batch = self.sess.run([self.loss], feed_dict=validate_dict)
        return loss_this_batch
    # def test(self, data_r, data_1, data_2):
    #
    #     data_r=np.array(data_r).reshape([self.batch_size,self.height,self.width,self.channel])
    #     data_1=np.array(data_1).reshape([self.batch_size,self.height,self.width,self.channel])
    #     data_2=np.array(data_2).reshape([self.batch_size,self.height,self.width,self.channel])
    #
    #     test_dict={self.image_r:data_r, self.image_1:data_1, self.image_2:data_2}
    #     prediction_this_batch,=self.sess.run([self.prediction,],feed_dict=test_dict)
    #     return prediction_this_batch,

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
