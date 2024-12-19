import math

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from mtcnn.box_utils import nms, _preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_first_stage(image, net, scale, threshold):
    """运行 P-Net，生成边界框，并进行非极大值抑制（NMS）。
    参数：
        image: 一个 PIL.Image 实例。
        net: 一个 PyTorch 的 nn.Module 实例，表示 P-Net。
        scale: 一个浮动数，
        用这个数值对图像的宽度和高度进行缩放。
        threshold: 一个浮动数，
        在从网络的预测结果生成边界框时，设定的面部概率阈值。

    返回：
        一个形状为 [n_boxes, 9] 的浮动型 NumPy 数组，
        包含边界框的得分和偏移（4 + 1 + 4）。
    """

    with torch.no_grad():
        # 缩放图像并将其转换为浮点数组
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')

        img = Variable(torch.FloatTensor(_preprocess(img)).to(device))
        output = net(img)
        probs = output[1].data.cpu().numpy()[0, 1, :, :]
        offsets = output[0].data.cpu().numpy()
        # probs：每个滑动窗口中出现人脸的概率
        #偏移：转换到真正的边界框

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.
    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.
    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    #应用P-Net在某种意义上相当于
    #移动12x12的窗口，步幅2
    stride = 2
    cell_size = 12

    # 可能有脸的盒子的索引
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # 边界框的变换
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net应用于缩放图像
    # 所以我们需要重新调整边界框
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])
    # 为什么要加一个？

    return bounding_boxes.T
