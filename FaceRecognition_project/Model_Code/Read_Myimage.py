"""
采集自己的人脸图像
1.使用摄像头来采集图像
"""
import cv2
import dlib
import os
import sys
import random

"""
将获取的本人图像 存放在该文件下面
"""
output_image = 'E:/DeepLearning/ComProject/Project/Project/My_project/main_faces/'
size = 28
if not os.path.exists(output_image):
    os.makedirs(output_image)

"""
将输入目录中的图像转换成minor_image中的图像
进行图像识别
改变图片的亮度与对比度
#使用dlib自带的frontal_face_detector作为我们的特征提取器
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
"""
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img
detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)


"""
获取50张自身图像
从摄像头读取照片
转为灰度图片
使用detector进行人脸检测
"""
index = 1
while True:
    if (index <= 2):
        print('Being processed picture %s' % index)
        success, img = camera.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = detector(gray_img, 1)

        for i, d in enumerate(dst):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            """
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            """

            face = img[x1:y1,x2:y2]
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size,size))
            cv2.imshow('image', face)
            cv2.imwrite(output_image+'/'+str(index)+'.jpg', face)
            index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break

