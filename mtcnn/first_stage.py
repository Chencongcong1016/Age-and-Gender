import math

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from mtcnn.box_utils import nms, _preprocess

#- **作用**：检查是否有可用的 GPU（CUDA 设备）。如果有 GPU 可用（`torch.cuda.is_available()` 返回 `True`），则将 `device` 设置为 `'cuda'`（GPU 设备）；如果没有 GPU 可用，则使用 CPU (`'cpu'`)。
#- **目的**：确保模型和数据在合适的设备上进行计算，通常为了提高性能，使用 GPU 进行深度学习任务。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_first_stage(image, net, scale, threshold):
    """ 第一级运行（运行 P-Net，生成边界框，并进行非极大值抑制（NMS）。)
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
        width, height = image.size      #   width, height, scale=>400,401,0.6
        # print("-" * 40+"  width, height, scale "+"-" * 40)  # 添加分隔线
        # print( width, height, scale)
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)    #向上取整到最近的整数： sw, sh=>240 241
        # print("-" * 40+"  sw, sh "+"-" * 40)  # 添加分隔线
        # print( sw, sh)  

        # print("-" * 40+"  image "+"-" * 40)  # 添加分隔线
        # print(image)    
        img = image.resize((sw, sh), Image.BILINEAR)        #  image=> <PIL.Image.Image image mode=RGB size=240x241 at 0x25EF0C77860>
        # print("-" * 40+"  image——Image.BILINEAR "+"-" * 40)  # 添加分隔线
        # print(img)

        img = np.asarray(img, 'float32')    #   将缩放后的图像转换为 NumPy 数组，类型为 `float32`，准备输入深度学习模型
        # print("-" * 40+"  image——float32 "+"-" * 40)  # 添加分隔线
        # print(img)

        #- `_preprocess(img)`：假设这是一个预处理函数，用于将图像数据转换为神经网络模型所需的格式（例如标准化，减去均值，除以标准差等）。
        #- `torch.FloatTensor()`：将图像数据转换为 PyTorch 的 `FloatTensor` 类型，适用于输入神经网络。
        #- `to(device)`：将数据传输到指定的设备（CPU 或 GPU）。
        img = Variable(torch.FloatTensor(_preprocess(img)).to(device))      #进行预处理


        #- **作用**：将预处理后的图像输入到 `PNet` 模型中进行前向传播计算，得到模型的输出。
        #- `output` 可能是一个包含多个值的元组，具体取决于模型的设计。在很多人脸检测任务中，输出通常包含：
        #- **分类得分**：表示图片中是否存在目标物体（比如人脸）的概率。
        #- **偏移量（offsets）**：表示目标物体相对于边界框的位移，用于回归任务，如检测框的修正。
        #- **目的**：获取模型的输出，用于后续的分析或处理。
        output = net(img)   #   把图像传入神经网络，得到模型对这张图像做出的判断结果。

        #- **作用**：
        #- `output[1]`：假设 `output` 是一个包含多个元素的元组（通常是分类概率和偏移量），这里取 `output[1]` 作为分类概率的部分。
        #- `.data`：获取张量的数据部分，去除计算图和梯度。
        #- `.cpu()`：将数据从 GPU 移动到 CPU。如果当前在 GPU 上运行，并且你需要将数据传递给 CPU 进行后续处理（如 NumPy 操作），就需要调用 `.cpu()`。
        #- `.numpy()`：将 PyTorch 张量转换为 NumPy 数组，以便在 CPU 上进一步操作。
        #- `[0, 1, :, :]`：这表示从概率张量中取出特定部分的数据（假设模型输出的是一个四维张量，其中第二个维度是类别，`1` 表示目标类别的概率）。
        #- **目的**：提取分类概率（这里是目标物体为类别 `1` 的概率）并将其转换为 NumPy 数组，用于后续处理。  
        probs = output[1].data.cpu().numpy()[0, 1, :, :]

        # 1. **`output[1]`**
        #`output` 可能是一个包含多个元素的容器（如列表或元组），其中 `output[1]` 是取 `output` 中的第二个元素（索引从0开始）。假设这个输出是一个包含多个部分的结果，其中：
        #- `output[0]` 可能是网络的其他信息（比如回归结果、位置数据等）。
        #- `output[1]` 可能是分类概率的输出或与人脸相关的概率值。

        #因此，`output[1]` 可能是一个 **多维数组**，通常是一个包含图像每个位置分类概率的 4D 张量。它的形状一般是 `(batch_size, num_classes, height, width)`，其中：
        #- `batch_size` 是输入图片的批次大小，通常是1。
        #- `num_classes` 是分类的类别数，通常是2（比如人脸和背景两个类别）。
        #- `height` 和 `width` 是输出特征图的空间维度（通常经过卷积和池化操作后，尺寸会缩小）。

        # 2. **`.data`**
        #`.data` 是 PyTorch 中用于获取张量（Tensor）数据的属性。它返回原始的数据，不包含梯度信息。这样做通常是为了节省内存或者做推理时，不需要梯度计算。

        # 3. **`.cpu()`**
        #`.cpu()` 是将数据从 GPU 内存转移到 CPU 内存。如果你使用的是 GPU 进行计算，PyTorch 会将张量保存在 GPU 上。`.cpu()` 就是将这些数据从 GPU 内存转移到 CPU 内存，方便后续的操作。因为有时你需要把数据从 GPU 转移到 CPU 才能进一步处理或保存。

        # 4. **`.numpy()`**
        #`.numpy()` 将 PyTorch 张量转换成 NumPy 数组。NumPy 是 Python 中用于科学计算的一个非常流行的库，它提供了高效的数组操作。通过 `.numpy()`，你可以将 PyTorch 张量转为 NumPy 数组，这样可以利用 NumPy 的功能进行后续的处理。

        # 5. **`[0, 1, :, :]`**
        #这是数组的索引操作。假设 `output[1]` 的形状是 `(batch_size, num_classes, height, width)`，`output[1].data.cpu().numpy()` 会返回一个形状为 `(batch_size, num_classes, height, width)` 的 NumPy 数组。

        #- `[0]` 取第一个图片（batch_size 为 1 时通常只有一张图片）。
        #- `[1]` 表示取第二个类别的数据，假设类别 `0` 代表背景，类别 `1` 代表目标（如人脸）。所以 `output[1][0, 1, :, :]` 取的是**类别 1（人脸）的概率分布**。
        #- `[:, :]` 表示保留图像的 **高度** 和 **宽度** 两个维度的所有数据，也就是获得完整的概率图。

        # 举个例子
        #假设你正在做一个人脸检测任务，网络的输出包含以下内容：
        #- `output[1]` 的形状为 `(1, 2, 32, 32)`，表示有 1 张图像，2 个类别（背景和人脸），输出图像的尺寸是 32x32。
        #- `output[1][0, 0, :, :]` 是背景的概率图，表示每个像素点属于背景的概率。
        #- `output[1][0, 1, :, :]` 是人脸的概率图，表示每个像素点属于人脸的概率。
        offsets = output[0].data.cpu().numpy()
        # print("-" * 40+"  offsets "+"-" * 40)  # 添加分隔线
        # print(offsets)  

        # probs：每个滑动窗口中出现人脸的概率
        #偏移：转换到真正的边界框
        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None
        
        #这里应用了 **非极大值抑制（NMS）** 算法，来从候选框中筛选出最合适的框。`nms` 函数的输入是一个边界框列表，通常是 `boxes[:, 0:5]`，即提取出每个候选框的前 5 个属性，通常是：
        #- `x1, y1`（左上角坐标）
        #- `x2, y2`（右下角坐标）
        #- `score`（得分）
        #overlap_threshold=0.5` 是 NMS 中的 **重叠阈值**，表示如果两个框的重叠区域超过 50%，那么 NMS 会保留得分较高的那个框，并丢弃另一个框。
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]


def  _generate_bboxes(probs, offsets, scale, threshold):
    """
    函数 `_generate_bboxes` 的作用是：在图像中生成可能包含人脸的边界框，并返回这些边界框的详细信息。

    ### 函数参数
    1. **`probs`**: 一个 `n` x `m` 的浮动 NumPy 数组，表示每个位置是否可能包含人脸的概率值。通常，值越大，说明当前位置包含人脸的可能性越高。
    2. **`offsets`**: 一个形状为 `[1, 4, n, m]` 的浮动 NumPy 数组，表示 P-Net 输出的 4 个偏移量，用来调整每个候选框的位置。它的形状通常包含 4 个元素，分别是 `(tx1, ty1, tx2, ty2)`，表示边界框的位置偏移。
    3. **`scale`**: 一个浮动数值，用于缩放图像的大小。由于 P-Net 通常是在缩放过的图像上运行，因此需要根据该 `scale` 对生成的边界框进行调整。
    4. **`threshold`**: 一个浮动数值，表示概率的阈值。只有超过这个概率值的候选区域才会被认为是潜在的人脸位置。

    #### 函数返回值
    - 返回一个形状为 `[n_boxes, 9]` 的 NumPy 数组，其中 `n_boxes` 是符合条件的边界框数量，9 表示每个边界框的 9 个属性：
    - `x1, y1` 是左上角坐标，
    - `x2, y2` 是右下角坐标，
    - `score` 是该边界框的分类分数（即是否是人脸的概率），
    - `tx1, ty1, tx2, ty2` 是四个偏移量，用于修正边界框。
    """

    #应用P-Net在某种意义上相当于
    #移动12x12的窗口，步幅2
    stride = 2      #是滑动窗口的步长，表示窗口每次向右或向下移动的像素数。
    cell_size = 12  #是 P-Net 使用的固定窗口大小，通常是 12x12。

    # 可能有脸的盒子的索引
    inds = np.where(probs > threshold)  #这里通过 `np.where` 找出 `probs` 数组中大于 `threshold` 的所有元素的索引，即可能包含人脸的区域的索引。

    if inds[0].size == 0:       #如果没有任何位置的概率值超过阈值 `threshold`，则返回一个空数组，表示没有检测到人脸。
        return np.array([])

    # 边界框的变换
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]    #这里从 `offsets` 数组中提取对应位置的偏移量（`tx1`, `ty1`, `tx2`, `ty2`）。这些偏移量会稍后用于调整候选框的位置。
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])        #将偏移量组合成一个数组，并提取对应位置的概率值作为得分 `score`，即这个位置属于人脸的可能性。
    score = probs[inds[0], inds[1]]

    # P-Net应用于缩放图像
    # 所以我们需要重新调整边界框
    #- 生成最终的边界框信息。`np.vstack` 用于按垂直方向堆叠数组。具体来说：
    #- `np.round((stride * inds[1] + 1.0) / scale)` 和类似的计算，用来确定边界框的左上角 (`x1, y1`) 和右下角 (`x2, y2`) 的位置。
    # `stride` 是步长，`inds` 给出了哪些位置满足阈值条件。
    # 计算时，`+1.0` 是为了对齐位置（因为在某些实现中，坐标是从1开始的）。
    #- 通过除以 `scale`，对位置进行缩放，以适应原始图像大小（P-Net 是在缩放后的图像上运行的）。
    #- 最后，`score` 和 `offsets` 会一并添加到边界框信息中，提供每个框的得分和相应的偏移量。
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale), 
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])
    # print("-" * 40+"  bounding_boxes[0] "+"-" * 40)  # 添加分隔线
    # print(bounding_boxes[0])  
    #   最终返回的是边界框的转置数组，每一行表示一个边界框，包括其坐标、得分和偏移量。
    return bounding_boxes.T
