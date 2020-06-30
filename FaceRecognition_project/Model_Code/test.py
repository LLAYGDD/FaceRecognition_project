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

main_faces='../main_faces'
minor_faces='../minor_faces'
if main_faces==0:
    print('main_faces:为空')
else:
    print('文件不为空')

"""
获取文件中的图像size
并对文件中的图像进行标签序列化
讲文件中所有图像进行归回处理
"""
size = 64
images = []
labels = []

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
# for lab in labels:
#     if lab==main_faces:
#         print("labels;=1")
#         labels=np.array([0,1])
#     else:
#         labels=np.array([1,0])
#         print("labels=0")
#labels = np.array([0 if label.endswith('main_faces') else 1 for label in labels])
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
                                                 test_size=0.05,random_state=0)
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
print(x)
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(shape=[None,2] , dtype=tf.float32)
#y_ = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=2)
print(y_)

keep_prob_one = tf.placeholder(tf.float32)
keep_prob_two = tf.placeholder(tf.float32)


"""
层数和参数的定义
conv、pooling、basis、weight
"""
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def dropout(x, keep):
    return tf.nn.dropout(x, keep)

"""
CNN层数的定义
经典三层模型
"""
def CNNLayer():
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_one)

    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_one)

    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_one)

    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_two)

    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def CNNTrain():
    out = CNNLayer()
    """
    交叉验证
    1.比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    2.将loss与accuracy保存以供tensorboard使用
    """
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    """
    训练
    交互接口 
    1.使用 summary将模型打印出来
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('../tmp', graph=tf.get_default_graph())
        for n in range(50):
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y,
                                                      keep_prob_one:0.5,
                                                      keep_prob_two:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, loss)
                print("loss:",loss)
                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_one:1.0, keep_prob_two:1.0})
                    print("acc:",acc)
                    print(n*num_batch+i, acc)
                    # 准确率大于0.98时保存并退出
                    if acc > 0.98 and n > 20:
                        saver.save(sess, 'model/train_faces.model')
                        #saver.save(sess, '../train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)
        print('accuracy > 0.98, exited!')
        saver.save(sess, 'model/train_faces.model')
        print('保存成功')

CNNTrain()

