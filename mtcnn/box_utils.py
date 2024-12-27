import numpy as np
from PIL import Image


def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.
    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """
    """ 非极大值抑制
    这个代码是一个 **非极大值抑制 (Non-Maximum Suppression, NMS)** 算法的实现，它常用于计算机视觉领域，特别是目标检测任务中。NMS 用于从多个候选框（bounding boxes）中选择出最优的框，抑制那些与其他框重叠较大的框。

    ### 解释代码和参数：

    #### 参数：
    1. **boxes**: 
         - 这是一个形状为 `[n, 5]` 的二维浮动数值 `numpy` 数组。每一行代表一个边界框（bounding box），包含以下五个元素：
         - `xmin`: 边界框左上角的 x 坐标。
         - `ymin`: 边界框左上角的 y 坐标。
         - `xmax`: 边界框右下角的 x 坐标。
         - `ymax`: 边界框右下角的 y 坐标。
         - `score`: 边界框的置信度分数，表示该框包含目标的概率。

    2. **overlap_threshold**:
       - 这是一个浮动数值，表示两个框之间允许的最大重叠比例。如果两个框的重叠比例（通常是计算交并比 IoU）超过这个阈值，那么重叠较小的框会被抑制（删除）。
       - 默认值为 0.5，意味着如果两个框的重叠区域大于 50%，则会被抑制。

    3. **mode**:
       - 这是一个字符串，表示在计算框之间的重叠时使用的模式。
       - `'union'`：计算交集和并集的比率，即使用交并比（IoU）来度量两个框的重叠程度。
       - `'min'`：计算两个框之间的最小重叠部分来度量它们的相似性。这种方式较少使用，但在一些情况下可能有用。

    #### 返回值：
    - 返回一个列表，包含被选择的框的索引。即从输入的 `boxes` 数组中，哪些框在经过非极大值抑制后被保留下来，哪些框被删除。

    ### 非极大值抑制 (NMS) 的工作原理：

    1. **排序**：首先，根据每个框的 `score`（置信度分数）对框进行排序，通常选择得分最高的框作为优先考虑的框。
    2. **选择**：选择得分最高的框，作为保留框。
    3. **重叠计算**：计算该框与其他框的重叠区域。如果该框与其他框的重叠区域（通过 IoU 计算）大于 `overlap_threshold`，则将重叠较大的框剔除。
    4. **迭代**：重复上述过程，直到所有框都被检查过。

    ### 示例：

    假设有三个框，它们的坐标和置信度如下：

    ```python
    boxes = np.array([
        [100, 100, 200, 200, 0.9],  # 框1: (xmin=100, ymin=100, xmax=200, ymax=200, score=0.9)
        [150, 150, 250, 250, 0.8],  # 框2: (xmin=150, ymin=150, xmax=250, ymax=250, score=0.8)
        [300, 300, 400, 400, 0.7],  # 框3: (xmin=300, ymin=300, xmax=400, ymax=400, score=0.7)
    ])
    ```

    - `boxes` 包含了三个框，分别有不同的置信度分数。
    - 如果设置 `overlap_threshold=0.5` 和 `mode='union'`，那么：
      - 首先选择得分最高的框（框1，`score=0.9`）。
      - 然后检查框1与框2之间的重叠程度，如果它们的 IoU 大于 0.5（即重叠超过 50%），则删除框2。
      - 框3与框1的重叠较小，可能会被保留。

    最终，返回的是选择框的索引（如 `[0, 2]`），表示框1和框3被保留。

    ### 代码实现示例：

    下面是一个简化版的 NMS 实现：

    ```python
    import numpy as np

    def nms(boxes, overlap_threshold=0.5, mode='union'):
        # 排序，按置信度从高到低
        scores = boxes[:, 4]
        indices = np.argsort(scores)[::-1]
    
        selected_indices = []

        while len(indices) > 0:
            # 选择得分最高的框
            current_index = indices[0]
            selected_indices.append(current_index)

            # 计算其他框与当前框的重叠区域
            current_box = boxes[current_index]
            other_boxes = boxes[indices[1:]]
        
            iou = compute_iou(current_box, other_boxes, mode)
        
            # 保留重叠度小于阈值的框
            indices = indices[1:][iou <= overlap_threshold]

        return selected_indices

    def compute_iou(box1, boxes, mode='union'):
        # 计算两个框之间的IoU
        x1, y1, x2, y2 = box1[:4]
        x1_other, y1_other, x2_other, y2_other = boxes[:, :4].T

        # 计算交集区域
        x1_inter = np.maximum(x1, x1_other)
        y1_inter = np.maximum(y1, y1_other)
        x2_inter = np.minimum(x2, x2_other)
        y2_inter = np.minimum(y2, y2_other)

        # 计算交集面积
        inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

        # 计算并集面积
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_other - x1_other) * (y2_other - y1_other)
    
        if mode == 'union':
            union_area = area1 + area2 - inter_area
        elif mode == 'min':
            union_area = np.minimum(area1, area2)
    
        # 计算IoU
        iou = inter_area / union_area
        return iou
    ```

    ### 总结：
    `nms` 函数实现了非极大值抑制，目的是根据边界框的得分和重叠程度筛选出最优框，减少冗余框。通过设置重叠阈值 `overlap_threshold` 和选择计算重叠方式的 `mode`，可以控制框的筛选策略。
    
    """

    # 如果没有方框，则返回空列表
    if len(boxes) == 0:
        return []

    # 所选索引列表
    pick = []

    # 获取边界框的坐标
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]  # range(5) 生成的是从 0 到 4 的数字
    """ boxes[:, i] 是一个切片操作切片操作,: 表示选取所有行（即每个边界框的所有数据），i 表示选择第 i 列（从 0 到 4）。
   
    切片语法是 list[start:end]，用于从 list 中选择一部分元素，其中：
    start 是起始索引（包含），默认为 0。
    end 是结束索引（不包含），如果省略，则默认为列表的最后一个元素。

    boxes[: i] 中
    : 表示从列表的开始位置开始。
     i 是结束位置（但不包括  i 位置的元素），表示从列表的第 0 个元素一直切片到第 last-1 个元素。


    例如，假设 boxes 如下所示：

    boxes = np.array([
        [100, 100, 200, 200, 0.9],  # 第 1 个边界框
        [150, 150, 250, 250, 0.8],  # 第 2 个边界框
        [300, 300, 400, 400, 0.7]   # 第 3 个边界框
    ])
    boxes[:, 0] 会返回 [100, 150, 300]，即所有边界框的 xmin 值。
    boxes[:, 1] 会返回 [100, 150, 300]，即所有边界框的 ymin 值。
    boxes[:, 2] 会返回 [200, 250, 400]，即所有边界框的 xmax 值。
    boxes[:, 3] 会返回 [200, 250, 400]，即所有边界框的 ymax 值。
    boxes[:, 4] 会返回 [0.9, 0.8, 0.7]，即所有边界框的置信度分数。
    """
    # print("-" * 40+"  x1, y1, x2, y2, score "+"-" * 40)  # 添加分隔线
    # print( x1, y1, x2, y2, score)

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # 按递增顺序排列；是从得分最低到得分最高的框的索引。
    """ 将会对数组中的数值进行排序，但返回的是排序后的索引位置，而不是排序后的数值本身。
    示例：
        score = np.array([0.9, 0.8, 0.95, 0.7, 0.85])
        ids = np.argsort(score)
        print(ids)
    
        - 0.7 (索引 3)
        - 0.8 (索引 1)
        - 0.85 (索引 4)
        - 0.9 (索引 0)
        - 0.95 (索引 2)
    
    输出结果为：
        [3 1 4 0 2]

    ### 结果说明：

    - `ids` 数组 `[3, 1, 4, 0, 2]` 表示原始 `score` 数组中每个元素的索引，在排序后它们的顺序变成了：
    - 排序后最小的元素 `0.7` 处于原数组的索引 `3` 位置。
    - 排序后的第二小的元素 `0.8` 处于原数组的索引 `1` 位置。
    - 排序后的第三小的元素 `0.85` 处于原数组的索引 `4` 位置。
    - 排序后的第四小的元素 `0.9` 处于原数组的索引 `0` 位置。
    - 排序后的最大元素 `0.95` 处于原数组的索引 `2` 位置。
    """

    while len(ids) > 0:

        # 抓取最大值的索引
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)   #把索引计入集合中

        #计算交点
        #分数最大的盒子的#
        #和其他盒子一起

        # 交叉框的左上角
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        # print("-" * 40+"  ix1 "+"-" * 40)  # 添加分隔线
        # print(ix1)
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        # print("-" * 40+"  iy1 "+"-" * 40)  # 添加分隔线
        # print(iy1)


        # 交叉框的右下角
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        # print("-" * 40+"  ix2 "+"-" * 40)  # 添加分隔线
        # print(iy1)
        iy2 = np.minimum(y2[i], y2[ids[:last]])
        # print("-" * 40+"  iy2 "+"-" * 40)  # 添加分隔线
        # print(iy1)

        # 交叉框的宽度和高度
        w = np.maximum(0.0, ix2 - ix1 + 1.0)    #右下角ix2-左上角ix1=宽度
        # print("-" * 40+"  w "+"-" * 40)  # 添加分隔线
        # print(w)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)    #右下角iy2-左上角iy1=高度
        # print("-" * 40+"  h "+"-" * 40)  # 添加分隔线
        # print(h)

        # 十字路口的地区
        inter = w * h   #计算出的重叠面积
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])    #重叠面积占百分比
        elif mode == 'union':
            # 交联（欠条）
            overlap = inter / (area[i] + area[ids[:last]] - inter)      #重叠面积占百分比

        # 删除所有重叠太大的框
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])  #overlap_threshold=0.5即如果两个框的交并比超过 50%
        )

    return pick


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].
    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # 这里发生的事情是这样的：
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # 下面是它的更简洁的形式

    # 补偿总是这样吗
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image.
    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.
    Returns:
        a float numpy array of shape [n, 3, size, size].
    """

    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')

        img_array = np.asarray(img, 'uint8')
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # 调整
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.
    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.
    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.
        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # “e”代表结束
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # 我们需要从图像中剪出一个盒子。
    # （x, y, ex, ey）是方框的修正坐标
    # 在图像中。
    # （dx, dy, edx, edy）为切割框的坐标
    # 从图片上看。
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # 如果盒子的右下角太右了
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # 如果框的右下角太低
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # 如果框的左上角太左
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # 如果框的左上角太高
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list


def _preprocess(img):
    """ 预处理后的图像已经做好了输入神经网络的准备，符合大多数深度学习框架（如 PyTorch）的输入要求。
    Preprocessing step before feeding the network.
    Arguments:
        img: a float numpy array of shape [h, w, c].
    Returns:
        a float numpy array of shape [1, c, h, w].
    
    """

    #   `transpose((2, 0, 1))` 会将通道维度 `c` 移到最前面，新的形状变为 `(c, h, w)`。
    #   这样，图像的形状会从 `(h, w, c)` 变成 `(c, h, w)`，符合许多深度学习框架（如 PyTorch）对于图像数据的要求。
    img = img.transpose((2, 0, 1))

    #   `np.expand_dims(img, 0)` 会将其扩展为 `(1, 3, 400, 300)`，增加了一个批次维度。
    img = np.expand_dims(img, 0)
    
    #   假设图像的像素值范围是 `[0, 255]`，通常我们需要将其缩放到一个更小的范围，通常是 `[0, 1]` 或 `[-1, 1]`，这有助于加速神经网络的训练并提高收敛速度。
    #   具体的归一化操作：
    #   - `img - 127.5`：首先将每个像素值减去 127.5，这样像素值的范围变成了 `[-127.5, 127.5]`。这个步骤的目的是将像素值的中心调整到 0。
    #   - 乘以 `0.0078125`（相当于 `1 / 128`）：这一步将像素值缩放到更小的范围，大致落在 `[-1, 1]`。原因是 `127.5 * 0.0078125 ≈ 1`，因此每个像素值被缩放到 `[-1, 1]` 范围内。
    img = (img - 127.5) * 0.0078125
    return img
