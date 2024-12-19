import numpy as np
import torch
from torch.autograd import Variable

from mtcnn.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn.first_stage import run_first_stage
from mtcnn.models import PNet, RNet, ONet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect_faces(image, min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
        参数：
        image：一个 PIL.Image 实例。
        min_face_size：一个浮动数值。
        thresholds：一个长度为 3 的列表。
        nms_thresholds：一个长度为 3 的列表。

        返回：
        两个浮动类型的 NumPy 数组，形状分别为 [n_boxes, 4] 和 [n_boxes, 10]，分别表示边界框和面部关键点。
    """

    # 禁用梯度计算，节省内存和计算资源
    with torch.no_grad():
        
        # 负荷模型  
        pnet = PNet().to(device)    # 初始化并加载 PNet 模型，将其移至指定设备（GPU 或 CPU）    
        rnet = RNet().to(device)    # 初始化并加载 RNet 模型，将其移至指定设备（GPU 或 CPU）     
        onet = ONet().to(device)    # 初始化并加载 ONet 模型，将其移至指定设备（GPU 或 CPU）        
        onet.eval()     # 将 ONet 模型设置为评估模式（例如：关闭 Dropout 层）

        # 建立一个图像金字塔
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # 缩放图像的尺度
        scales = []

        # 缩放图像，以便
        # 我们可以检测到的最小尺寸等于
        # 我们想要检测的最小面大小
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:          # 当图像的最小边长大于设定的最小检测尺寸时，继续循环
            scales.append(m * factor ** factor_count)        # 计算当前尺度并添加到 scales 列表中
            min_length *= factor                # 更新 min_length，乘以比例因子 factor
            factor_count += 1           # 增加比例计数器

        # 阶段1

        # 它会被退回
        bounding_boxes = []

        # 在不同规模上运行P-Net
        for s in scales:
            boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)    # 将当前数组boxes添加到 bounding_boxes 列表中

        # 收集不同尺度的方框（以及偏移量和分数）
        bounding_boxes = [i for i in bounding_boxes if i is not None]       # 从 bounding_boxes 中筛选出非 None 的元素。
        bounding_boxes = np.vstack(bounding_boxes)      # 将过滤后的有效边界框列表垂直堆叠（即将多个数组按行连接成一个二维数组）

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # 使用pnet预测的偏移量来转换边界框
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # 第二阶段

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes).to(device))
        output = rnet(img_boxes)
        offsets = output[0].data.cpu().numpy()  # shape [n_boxes, 4]
        probs = output[1].data.cpu().numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # 第三阶段

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes).to(device))
        output = onet(img_boxes)
        landmarks = output[0].data.cpu().numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.cpu().numpy()  # shape [n_boxes, 4]
        probs = output[2].data.cpu().numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # 计算地标点
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks
