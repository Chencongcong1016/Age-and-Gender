import cv2 as cv
import numpy as np
from PIL import Image

from mtcnn.detector import detect_faces
from mtcnn.visualization_utils import show_bboxes

if __name__ == '__main__':
    img = Image.open('images/office1.jpg')                  # 打开第一张图像，路径为 'images/office1.jpg'
    bounding_boxes, landmarks = detect_faces(img)           # 使用 MTCNN 人脸检测函数检测图片中的人脸框和人脸特征点（如眼睛、鼻子等）
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)      # 将图像从 RGB 格式转换为 OpenCV 所使用的 BGR 格式
    show_bboxes(img, bounding_boxes, landmarks)             # 调用函数显示图像，带上检测到的人脸框和特征点

    img = Image.open('images/office2.jpg')                  # 打开第二张图像，路径为 'images/office2.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office3.jpg')                  # 打开第三张图像，路径为 'images/office3.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office4.jpg')                  # 打开第四张图像，路径为 'images/office4.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office5.jpg')                  # 打开第五张图像，路径为 'images/office5.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)
