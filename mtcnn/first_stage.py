import math

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from mtcnn.box_utils import nms, _preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_first_stage(image, net, scale, threshold):
    """Run P-Net, generate bounding boxes, and do NMS.
    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.
    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
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
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes

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
