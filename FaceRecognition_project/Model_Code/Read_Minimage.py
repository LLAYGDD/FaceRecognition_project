"""
将文件中的所有图像
裁剪并保存在固定的文件中
"""
import sys
import os
import cv2
import dlib

input_image = 'E:/DeepLearning/ComProject/Project/Project/My_project/input_img/'
output_image = 'E:/DeepLearning/ComProject/Project/Project/My_project/minor_faces/'
#判断文件夹是否存在
if not os.path.exists(output_image):
    os.makedirs(output_image)
"""
使用dlib自带的frontal_face_detector作为我们的特征提取器
"""
size = 64
detector = dlib.get_frontal_face_detector()
"""
将输入目录中的图像转换成minor_image中的图像
进行图像识别
"""
index = 1
for (path, dirnames, filenames) in os.walk(input_image):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            filename = path+'/'+filename
            """
            # 从文件读取图片
            # 转为灰度图片
            # 使用detector进行人脸检测 dets为返回的结果
            """
            img = cv2.imread(filename)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dst = detector(gray_img, 1)

            """
            #使用enumerate 函数遍历序列中的元素以及它们的下标
            #下标i即为人脸序号
            #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
            #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
            # img[y:y+h,x:x+w]
            # 调整图片的尺寸
            # 保存图片
            """
            for i, d in enumerate(dst):
                """
                图像中人脸位置的检测和表示
                """
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1,x2:y2]
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                cv2.imwrite(output_image+'/'+str(index)+'.jpg', face)
                index += 1
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)

