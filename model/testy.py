import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from keras.datasets import cifar10


def preprocess(img,label):
    label = label[:3000].reshape((len(label[:3000],)))
    img = img / 255.0
    lenth = len(img[:3000,:,:,:])

    imag_list = []
    for i in range(lenth):
        resized_img = skimage.transform.resize(img[i,:,:,:], (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
        imag_list.append(resized_img)
    imag_list = np.reshape(imag_list,[-1,224,224,3])
    return imag_list,label


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.tfx = tf.placeholder(tf.float32, [20, 224, 224, 3])
        self.tfy = tf.placeholder(tf.int32, [20,])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        W1 = tf.get_variable(name='W1',shape=[7*7*512,256],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1', shape=[256, ],initializer=tf.truncated_normal_initializer(stddev=0.2))
        x = tf.layers.batch_normalization(tf.add(tf.matmul(self.flatten, W1), b1))

        W2 = tf.get_variable(name='W2', shape=[256,64], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name='b2', shape=[64, ], initializer=tf.truncated_normal_initializer(stddev=0.2))
        self.out = tf.add(tf.matmul(x, W2), b2)
        self.prediction = tf.nn.softmax(self.out)


        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            #self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.tfy))
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    def predict(self, x,y):
        prediction = self.sess.run(self.prediction,{self.tfx: x, self.tfy: y})
        return prediction



    def save(self, path='./for_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train,y_train = preprocess(x_train,y_train)
    vgg = Vgg16(vgg16_npy_path='pretrained_model/vgg16.npy')
    print('Net built')
    for i in range(300):
        b_idx = np.random.randint(0, len(x_train), 20)
        train_loss = vgg.train(x_train[b_idx], y_train[b_idx])
        print(i, 'train loss: ', train_loss)
        vgg.save('model_saved/transfer_learn'+'%d'%(i))      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='pretrained_model/vgg16.npy',
                restore_from='model_saved/transfer_learn296')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test, y_test = preprocess(x_test, y_test)
    accuracy = 0
    for i in range(100):
        b_idx = np.random.randint(0, len(x_test), 20)
        predict = vgg.predict(x_test[b_idx], y_test[b_idx])
        predict = np.argmax(predict, axis=1)
        k = y_test[b_idx]
        count = 0
        for j in range(20):
            if k[j]==predict[j]:
                count+=1
        accuracy += count
        c=accuracy/(20.0+20*i)
        print(count," ",c)
    acc = accuracy/len(y_test)

if __name__ == '__main__':

    # download()
    #train()
    eval()