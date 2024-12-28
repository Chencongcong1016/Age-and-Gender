import cv2 as cv
import numpy as np
from PIL import Image

from detector import detect_faces
from visualization_utils import show_bboxes

if __name__ == '__main__':
    img = Image.open('images/0_raw.jpg')                  # 打开第一张图像，路径为 'images/0_raw.jpg'
    bounding_boxes, landmarks = detect_faces(img)           # 使用 MTCNN 人脸检测函数检测图片中的人脸框和人脸特征点（如眼睛、鼻子等）
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)      # 将图像从 RGB 格式转换为 OpenCV 所使用的 BGR 格式
    show_bboxes(img, bounding_boxes, landmarks)             # 调用函数显示图像，带上检测到的人脸框和特征点

    img = Image.open('images/1_raw.jpg')                  # 打开第二张图像，路径为 'images/1_raw.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/2_raw.jpg')                  # 打开第三张图像，路径为 'images/2_raw.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/3_raw.jpg')                  # 打开第四张图像，路径为 'images/3_raw.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/4_raw.jpg')                  # 打开第五张图像，路径为 'images/4_raw.jpg'
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)
