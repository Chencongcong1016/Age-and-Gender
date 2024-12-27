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
        
        # 负荷模型  `PNet` 是一个神经网络模型（通常是卷积神经网络），它用来做图像分类或目标检测任务。
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

        # 调试打印：bounding_boxes
            # print("-" * 40+" image "+"-" * 40)  # 添加分隔线
            """ Python 中通过 PIL（Pillow）库表示的一个图像对象的字符串表示
            `<PIL.Image.Image image mode=RGB size=400x401 at 0x270CAC64B00>` 是 Python 中通过 PIL（Pillow）库表示的一个图像对象的字符串表示。
            让我们逐部分分析一下这个内容：
            1. **`PIL.Image.Image`**：表示该对象是一个图像对象，属于 `PIL.Image` 类。PIL（Python Imaging Library）是一个用于处理图像的库，`Image` 是其中的一个类，用于表示图像数据。
            
            2. **`image mode=RGB`**：表示该图像的颜色模式是 RGB（红色、绿色、蓝色）。这种模式通常用于彩色图像，每个像素由红、绿、蓝三种颜色组成。
            
            3. **`size=400x401`**：表示图像的尺寸是 400 像素宽，401 像素高。也就是说，该图像的宽度为 400 像素，高度为 401 像素。
            
            4. **`at 0x270CAC64B00`**：表示该对象在内存中的地址，即这个图像对象的存储位置。这是 Python 对象在内存中的标识符。
            
            总结起来，`<PIL.Image.Image image mode=RGB size=400x401 at 0x270CAC64B00>` 这个字符串表示的是一个 RGB 彩色图像，尺寸为 400x401 像素，且该图像对象位于内存中的某个位置（`0x270CAC64B00`）。
            这个信息一般是由 Python 的解释器或调试工具自动生成的，用来描述对象的类型、属性和内存位置。
            
            """

            # print(image)      #<PIL.Image.Image image mode=RGB size=400x401 at 0x270CAC64B00>

            # print("-" * 40+" pnet "+"-" * 40)  # 添加分隔线
            """ PNet 是广泛用于人脸检测的多阶段网络之一
            这是一个 PyTorch 模型的定义，表示了一个名为 `PNet` 的神经网络结构。这个结构可能是用来进行人脸检测（例如，PNet 是广泛用于人脸检测的多阶段网络之一）。我们可以逐层分析该网络的组件。

            ### 1. **`PNet`**
            `PNet` 可能是一个面向特定任务的神经网络（如人脸检测），该网络由多个层组成，分为几个部分：特征提取部分（`features`）和最后的卷积层（`conv4_1` 和 `conv4_2`）。

            ### 2. **`features`（特征提取部分）**
            `features` 是一个 `Sequential` 容器，包含了多个卷积层（`Conv2d`）和 PReLU 激活函数层（`PReLU`），以及一个池化层（`MaxPool2d`）。这些层一起用于从输入图像中提取特征。

            - **`conv1`**: `Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))`  
              第一个卷积层，将输入的 3 通道 RGB 图像（3 个通道）转换为 10 个特征图，卷积核大小为 3x3，步长为 1x1。

            - **`prelu1`**: `PReLU(num_parameters=10)`  
              激活函数，PReLU（Parametric ReLU）是 ReLU 的一种变体，允许每个通道使用不同的参数来控制激活。

            - **`pool1`**: `MaxPool2d(kernel_size=2, stride=2)`  
              池化层，采用 2x2 的池化窗口和步长 2，目的是减少特征图的空间维度（下采样）。

            - **`conv2`**: `Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1))`  
              第二个卷积层，将前一层的 10 个特征图转换为 16 个特征图，卷积核仍为 3x3，步长为 1x1。

            - **`prelu2`**: `PReLU(num_parameters=16)`  
              第二个 PReLU 激活函数，作用于 16 个特征图。

            - **`conv3`**: `Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))`  
              第三个卷积层，将 16 个特征图转换为 32 个特征图，卷积核大小为 3x3，步长为 1x1。

            - **`prelu3`**: `PReLU(num_parameters=32)`  
              第三个 PReLU 激活函数，作用于 32 个特征图。

            ### 3. **`conv4_1` 和 `conv4_2`**
            这两层卷积层是在 `features` 部分之后，用于生成最终的预测结果。

            - **`conv4_1`**: `Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1))`  
              这是一个 1x1 的卷积层，将 32 个特征图转换为 2 个特征图。它可能用于进行某种二分类任务（例如，人脸检测中的目标和非目标分类）。

            - **`conv4_2`**: `Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))`  
              另一个 1x1 的卷积层，将 32 个特征图转换为 4 个特征图。这个层的输出通常用于回归任务（例如，框的边界框回归）。

            ### 总结
            这个网络的结构如下：

            - 输入图像（通常是 3 通道的 RGB 图像）经过一系列的卷积层（`conv1`, `conv2`, `conv3`）和激活函数（`prelu1`, `prelu2`, `prelu3`），通过池化层（`pool1`）逐步提取特征。
            - 最后，网络通过两个 1x1 的卷积层（`conv4_1`, `conv4_2`）输出预测结果。
              - `conv4_1` 输出 2 个特征图，可能用于分类（目标/非目标）。
              - `conv4_2` 输出 4 个特征图，可能用于回归（例如，检测框的坐标）。

            这个网络的设计显然是为某种目标检测任务（如人脸检测）服务的。
            """
           
            # print(pnet)      

            # print("-" * 40+" boxes "+"-" * 40)  # 添加分隔线
            """ boxes是一个目标检测模型的输出
            你提供的这个数据看起来是目标检测（例如人脸检测）模型的输出结果，特别像是某种类型的边界框（bounding box）检测的输出，通常用于物体检测任务中。

            具体来说，每一行的数据似乎代表了一个检测到的目标的预测信息。我们可以逐列解析这些值：

            1. **边界框的坐标**（第一列到第四列）：
               - **第一列：** 左上角的 x 坐标（通常是框的左侧）。
               - **第二列：** 左上角的 y 坐标（通常是框的上方）。
               - **第三列：** 右下角的 x 坐标（通常是框的右侧）。
               - **第四列：** 右下角的 y 坐标（通常是框的下方）。
   
               这些值描述了检测到目标的矩形边界框的位置。

            2. **置信度分数**（第五列）：
               - **第五列：** 检测到该目标的置信度分数（通常是一个值在 0 到 1 之间）。例如，`9.83116329e-01` 表示 98.31%的置信度，意味着模型非常确定它检测到的物体是一个目标。

            3. **类别得分和其他预测信息**（第六到第九列）：
               - **第六列到第九列：** 这些通常是类别预测得分或者边界框的回归调整参数（例如，边界框的位置偏移或缩放因子）。它们可能是类别标签的概率分布，或者与目标位置相关的额外信息（例如，目标的位置调整）。

            ### 举个例子：
            ```
            [ 7.20000000e+01  8.80000000e+01  9.20000000e+01  1.08000000e+02
               9.83116329e-01  8.66156258e-03 -5.73291704e-02 -1.22976720e-01
               3.02913487e-02]
            ```
            - 检测到的目标边界框是从 `(72, 88)` 到 `(92, 108)`。
            - 目标置信度分数为 `0.983`，表示模型认为这是一个有效目标的概率为 98.3%。
            - 其余的列是类别得分或与目标位置调整相关的预测信息。

            ### 总结：
            这组数据是一个目标检测模型的输出，其中包括了每个检测目标的边界框位置、置信度得分以及其他可能的额外预测信息。
            """

            # print(boxes)      
        # print("-" * 40+" bounding_boxes[0] "+"-" * 40)  # 添加分隔线
        # print(bounding_boxes[0])     #输出模型集合.它是一个列表 (list)，包含了两个 numpy 数组。


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
