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
    """ 检测面部
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
            
        pnet = PNet().to(device)          # 负荷模型  `PNet` 是一个神经网络模型（通常是卷积神经网络），它用来从图像中得到初步的检测框，筛选出可能包含人脸的区域。
        rnet = RNet().to(device)          # 神经网络（R-Net）进一步对人脸候选框进行筛选、校正和优化。
        onet = ONet().to(device)          # 这一阶段是在进行人脸检测（或者其他类型的目标检测）时，执行了基于候选框（bounding boxes）进行进一步的目标识别、关键点回归和分类。
        onet.eval()                       # 将 ONet 模型设置为评估模式（例如：关闭 Dropout 层）

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

        # 阶段1 ，它的作用： 
          # 这个阶段主要是通过**P-Net**（即**Proposal Network**）生成初步的候选框，并通过一些筛选步骤（比如非极大值抑制 NMS、回归修正等）得到初步的检测框。
          # 这个过程通常是人脸检测中第一阶段，用来从图像中筛选出可能包含人脸的区域。
     
        bounding_boxes = []                                                              # 初始化一个空的bounding_boxes列表来保存所有候选框

        # 在不同规模上运行P-Net
        for s in scales:
            boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])       # 对每个尺度进行处理，`run_first_stage()` 运行P-Net
            bounding_boxes.append(boxes) #添加到列表

        bounding_boxes = [i for i in bounding_boxes if i is not None]       # 去除空的bounding_boxes（可能某些尺度没有找到任何框）
        bounding_boxes = np.vstack(bounding_boxes)                          # 将所有候选框堆叠到一起，形成一个统一的数组
        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])               # 进行非极大值抑制（NMS），去掉重叠较大的框，只保留最有可能的框
        # print("-" * 40+" keep "+"-" * 40)  # 添加分隔线
        # print(keep)    
        bounding_boxes = bounding_boxes[keep]

        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])    # 对框进行回归修正，使用先前得到的偏移量对框进行细化

        bounding_boxes = convert_to_square(bounding_boxes)                  # 将候选框调整为正方形
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])           # 四舍五入框的坐标，使其变成整数

        # 第二阶段，它的作用
          # 这个阶段的目的是通过第二阶段的神经网络（R-Net）进一步对人脸候选框进行筛选、校正和优化。
          # 首先从图像中提取候选框，然后通过分类和回归预测出每个框的可能性和位置偏移量，
          # 再通过非极大值抑制去除重复框，
          # 最后调整框的形状和位置，确保框准确地包围目标。

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)     # 1. 将原始图像中所有候选框区域裁剪成统一大小的图像块，便于网络输入
        img_boxes = Variable(torch.FloatTensor(img_boxes).to(device))   # 2. 将裁剪后的图像转换为 Tensor 类型，并转移到计算设备（如 GPU）
        output = rnet(img_boxes)                                        # 3. 将图像输入到第二阶段深度神经网络 R-Net 中，得到输出
        offsets = output[0].data.cpu().numpy()                          # 4. 从输出中提取偏移量和概率
        probs = output[1].data.cpu().numpy()

        keep = np.where(probs[:, 1] > thresholds[1])[0]                 # 5. 筛选掉那些概率不高的框，只保留那些认为是人脸的框
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])                   # 6. 使用 NMS（非极大值抑制）去除重叠过多的框，保留最合适的框
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])   # 7. 使用偏移量调整框的位置，使其更加精确
        bounding_boxes = convert_to_square(bounding_boxes)              # 8. 将边界框调整为正方形
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])       # 9. 将边界框的坐标四舍五入为整数，确保它们是有效的像素坐标

        # 第三阶段，它的作用
          # 这一阶段是在进行人脸检测（或者其他类型的目标检测）时，执行了基于候选框（bounding boxes）进行进一步的目标识别、关键点回归和分类。

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)     # 将原始图像中所有候选框区域裁剪成统一大小的图像块(这里设定为 48x48)，便于网络输入
        if len(img_boxes) == 0:                                         # 安全检查，确保后续步骤不会对空数据进行操作
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes).to(device))   # 将裁剪后的图像转换为 Tensor 类型，并转移到计算设备（如 GPU）
        output = onet(img_boxes)                                        # 将图像输入到第三阶段深度神经网络 ONet 中，得到输出
        landmarks = output[0].data.cpu().numpy()                        # 关键点坐标（例如，5个人脸关键点，每个关键点有 2 个坐标，因此是形状 `[n_boxes, 10]`）。
        offsets = output[1].data.cpu().numpy()                          # 边界框的偏移量（通常用于微调检测框的位置，是形状 `[n_boxes, 4]`）。
        probs = output[2].data.cpu().numpy()                            # 分类概率（通常是一个二分类结果，表示该框是否包含目标，是形状 `[n_boxes, 2]`）。

        keep = np.where(probs[:, 1] > thresholds[2])[0]                 # 筛选掉那些概率不高的框，只保留那些认为是人脸的框
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]                                         # 保留与筛选后的边界框相关的偏移量和关键点坐标。
        landmarks = landmarks[keep]

        # 计算地标点
          # 这段代码涉及目标检测中地标点（landmarks）和边界框（bounding boxes）的处理，
          # 通常在物体检测任务中，除了对目标的边界框进行预测，还会同时预测一些地标点（如人脸的眼睛、鼻子、嘴巴的位置等）。
          # 这里的代码通过给定的边界框对地标点进行校正，
          # 然后应用非极大值抑制（NMS）来去除重叠框。下面我将逐行注释并解释每一部分的含义，并通过一个例子来加深理解。

        # 计算边界框的宽度和高度
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0

        # 获取边界框的左上角坐标
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]

        # 调整地标点的坐标 (基于边界框的宽度和高度)
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)       # 对边界框进行校正，考虑偏移量
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')     # 使用非极大值抑制 (NMS) 去除重叠框
        bounding_boxes = bounding_boxes[keep]                         # 保留最终的边界框
        landmarks = landmarks[keep]                                   # 保留最终的地标点

        return bounding_boxes, landmarks
