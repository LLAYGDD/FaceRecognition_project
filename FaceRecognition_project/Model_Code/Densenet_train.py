"""
数据包的创建和导入
"""
import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm,flatten
from tensorflow.contrib.framework import arg_scope



"""
层数和参数的定义
conv、pooling、basis、weight
基本的包的导入和创建
"""
growth_k = 12
nb_block = 2 
init_learning_rate = 1e-4
epsilon = 1e-8 
dropout_rate = 0.2
nesterov_momentum = 0.9
weight_decay = 1e-4
class_num = 2
total_epochs = 50

"""
测试数据的输入
1.获取文件中的图像size
2.并对文件中的图像进行标签序列化
3.讲文件中所有图像进行归回处理

"""


size = 64
images = []
labels = []

main_faces='E:/DeepLearning/ComProject/Project/Project/My_project/main_faces'
minor_faces='E:/DeepLearning/ComProject/Project/Project/My_project//minor_faces'
if main_faces==0:
    print('main_faces:为空')
else:
    print('文件不为空')

def getPaddingSize(img):
    h, w, _ = img.shape
    #初始化图像中人脸的四个坐标
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)
    #判断w和h
    if w < longest:
        tmp = longest - w
        """
        获取图像中人脸的位置
        左右位置
        left和right
        //表示整除符号
        """
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

"""
读入数据imagedata
"""
def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top,bottom,left,right = getPaddingSize(img)
            """
            为了提取图像中人脸的边缘讲图像整体放大并补充边缘
            Padding
            并将图像灰度化处理cv2.border
            将图片放大， 扩充图片边缘部分
            """
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))
            print("h",h)
            print("w",w)
            images.append(img)
            labels.append(path)


"""
图像路径的输入
将图像和标签转化成对应的数组array

"""
readData(main_faces)
readData(minor_faces)
images = np.array(images)
print(images.shape)

labels = np.array([[0,1] if lab == main_faces else [1,0] for lab in labels])
print(labels)

"""
模型训练和创建
目前使用cnn模型进行测试
后期改成动态路由
胶囊网络
1.随机划分测试集与训练集
2.参数：图片数据的总数，图片的高、宽、通道
3.# 将数据转换成小于1的数
"""

train_x,test_x,train_y,test_y = train_test_split(images,
                                                 labels,
                                                 test_size=0.05,
                                                 random_state=0)
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
"""
图像的类型的转化
reshape
转化成三通道rgb
将图像数据归一化处理
"""
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

# 图片块，每次取100张图片
batch_size = 32
num_batch = len(train_x) // batch_size
x = tf.placeholder(tf.float32, [None, 4096])
x = tf.placeholder(tf.float32, [None, size, size, 3])
#print(x)

batch_images = tf.reshape(x, [-1, size, size, 3])
label = tf.placeholder(tf.float32, shape=[None, 2])
print(label)


"""
层数和基本参数的定义
经典模式
"""
def conv_layer(input,filter,kernel,stride=1,layer_name="conv"):
    with tf.name_scope(layer_name):
        network=tf.layers.conv2d(inputs=input,
                                 filters=filter,
                                 kernel_size=kernel,
                                 strides=stride,
                                 padding='SAME')
        return network
"""
全局池化层的定义和创建
"""
def Global_Average_Pooling(x, stride=1):
    """
    图像的行和列数据集合
    width=np.shape(x)[1]
    height=np.shape(x)[2]
    :param x:
    :param stride:
    :return:
    下面使用h5对其进行相应的存储
    """
    return global_avg_pool(x,name='Global_Average_Pooling')

"""
图像归一化处理
"""
def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
"""
dropout的定义和创建
激活函数relu创建
"""

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)
def Relu(x):
    return tf.nn.relu(x)
def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x,
                                       pool_size=pool_size,
                                       strides=stride,
                                       padding=padding)
def Max_Pooling(x, pool_size=[3,3],
                stride=2,
                padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x,
                                   pool_size=pool_size,
                                   strides=stride,
                                   padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



"""
密集连接块和模型的创建

"""
class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
            x = Concatenation(layers_concat)
            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)

        """
        密集连接块的
        3个密集连接块
        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        #层数的添加
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)
        return x


training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

"""
优化器
损失函数
正则化定义和创建
"""

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
saver = tf.train.Saver(tf.global_variables())

print("程序结束")

"""
模型的训练
"""



with tf.Session() as sess:

    print("sesson")
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('../tmp', graph=tf.get_default_graph())
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('./logs', sess.graph)
    print("www")


    global_step = 0
    epoch_learning_rate = init_learning_rate

    for epoch in range(total_epochs):
        if epoch==(total_epochs*0.5) or epoch==(total_epochs*0.75):

            #学习率随着迭代次数的变化而调整
            epoch_learning_rate=epoch_learning_rate/10

            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                #print("ssss")

                #训练批次
                train_feed_dict = {x: batch_x,
                                   label: batch_y,
                                   learning_rate: epoch_learning_rate,
                                   training_flag : True}
                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)


                #print("cccc")
                if (epoch*num_batch+i) % 100 == 0:
                    global_step += 100
                    #print("10000")
                    train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                    print("Step:", epoch*num_batch+i, "Loss:", loss, "Training accuracy:", train_accuracy)
                    #print("nihao dao muqian ")

                test_feed_dict = {

                    x: test_x,
                    label: test_y,
                    learning_rate: epoch_learning_rate,
                    training_flag : False
                }
                print("lllll")

                accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
                print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)



