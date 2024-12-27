import datetime
import os
import pickle
import tarfile

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm

from config import IMG_DIR, pickle_file
from mtcnn.detector import detect_faces


def extract(filename):
    print('提取 {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


def reformat_date(mat_date): 
    """    重新格式化给定日期（mat_date），并在对可能的闰年进行调整后返回年份。此函数接受顺序格式的日期（即，从0001-01-01开始的天数），从中减去366天（以调整闰年），并返回结果日期所在的年份。
    重新格式化给定日期（mat_date），并在对可能的闰年进行调整后返回年份。
    此函数接受顺序格式的日期（即，从0001-01-01开始的天数），
    从中减去366天（以调整闰年），并返回结果日期所在的年份。

    参数:
    mat_date (int)：以顺序格式输入的日期（从0001-01-01开始的天数）。

    返回:
    int：日期减去366天后的年份。

    例子:
    > > > reformat_date (737900)
    2023
    """
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt


def create_path(path):
    """ 根据给定的路径（path）和图片目录（IMG_DIR），创建并返回一个正确格式的路径。
    根据给定的路径（path）和图片目录（IMG_DIR），
    创建并返回一个正确格式的路径。

    该函数将传入的路径（path）的第一个元素（path[0]）与全局变量 `IMG_DIR` 拼接，
    然后将路径中的反斜杠 ('\\') 替换为正斜杠 ('/')，以确保路径格式正确。

    参数：
    path (str): 输入的路径字符串，通常是一个相对路径或文件名。

    返回：
    str: 拼接后的路径字符串，确保使用正斜杠，并与 `IMG_DIR` 合并。

    例子：
    >>> create_path("example.jpg")
    "/home/user/images/example.jpg"  # 假设 IMG_DIR = "/home/user/images"
    """
    return os.path.join(IMG_DIR, path[0]).replace('\\','/')


def get_face_attributes(full_path):
    """ 获取人脸属性

    返回 False 和 None，表示没有有效的人脸检测
    """
    try:
        # 使用 PIL 打开图像文件，并将图像转换为 RGB 模式
        img = Image.open(full_path).convert('RGB')

        # 调用 detect_faces 函数检测图像中的人脸
        # bounding_boxes 是人脸的边界框坐标，landmarks 是人脸关键点
        bounding_boxes, landmarks = detect_faces(img)

        # 获取图像的宽度和高度
        width, height = img.size

         # 如果检测到图像中只有一张人脸
        if len(bounding_boxes) == 1:
            # 提取该人脸的边界框坐标（左上角和右下角的坐标）
            x1, y1, x2, y2 = bounding_boxes[0][0], bounding_boxes[0][1], bounding_boxes[0][2], bounding_boxes[0][3]
            # 检查边界框是否有效：坐标值是否在图像范围内，且左上角坐标不能大于右下角坐标
            if x1 < 0 or x1 >= width or x2 < 0 or x2 >= width or y1 < 0 or y1 >= height or y2 < 0 or y2 >= height or x1 >= x2 or y1 >= y2:
                # 如果无效，返回 False 和 None
                return False, None, None

            # 将人脸关键点的坐标进行四舍五入并转化为整数
            landmarks = [int(round(x)) for x in landmarks[0]]

            # 检查边界框的大小是否超过图像的十分之一，确保检测到的人脸区域足够大
            is_valid = (x2 - x1) > width / 10 and (y2 - y1) > height / 10

            # 如果边界框足够大，返回有效的结果
            return is_valid, (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), landmarks

    except KeyboardInterrupt as e:
        # 如果程序被用户中断（如按下 Ctrl+C），抛出异常
        print("报错地方KeyboardInterrupt:", e)
        raise
    except Exception as e:
        # 其他异常情况下，返回 False 和 None
        print("报错地方Exception:", e)
        pass

    # 默认返回 False 和 None，表示没有有效的人脸检测
    return False, None, None


if __name__ == "__main__":
    if not os.path.isdir('data/imdb_crop'):
        extract('data/imdb_crop.tar')
    if not os.path.isdir('data/imdb'):
        extract('data/imdb_meta.tar')

    # 使用 scipy 库中的 io 模块来加载 .mat 文件的代码。
    mat = scipy.io.loadmat('data/imdb/imdb.mat')
    imdb = mat['imdb'][0, 0]
    """ 调试打印：imdb
    # 调试打印：imdb
    # print("------------------------------------------------imdb------------------------------------------------")
    # print(imdb)
    # for d in imdb:
    #     print("------------------------------------------------d------------------------------------------------")
    #     print(d)
    #     print("------------------------------------------------d[0]------------------------------------------------")
    #     print(d[0])
    #     data=d[0]
    #     print("------------------------------------------------data------------------------------------------------")
    #     print(data)
    """

    data = [d[0] for d in imdb]
    keys = ['dob',      # 出生日期
            'photo_taken',  # 照片拍摄年份
            'full_path',    # 文件路径
            'gender',   # 女性 0 个，男性 1 个，未知时为 NaN
            'name',     #名人姓名
            'face_location',    # 面部的位置
            'face_score',   # detector 分数（越高越好）
            'second_face_score',    # 得分第二高的人脸的 detector 分数
            'celeb_names',      # 所有名人姓名列表
            'celeb_id'      # 名人姓名索引
            ]
    
    # np.asarray(data, dtype=object)，是将 data 转换为 NumPy 数组，并确保数组的数据类型是 object
    # zip 函数是一个内建函数：如果 keys = ['a', 'b', 'c'] 和 data = [1, 2, 3]，那么 zip(keys, data) 会返回一个迭代器，生成如下的元组：('a', 1), ('b', 2), ('c', 3)。
    imdb_dict = dict(zip(keys, np.asarray(data, dtype=object))) 
    """ 调试打印：imdb_dict
    # 调试打印：imdb_dict
    # for key, value in imdb_dict.items():
    #     print(f"Key: {key}")
    #     print(f"Value: {value}")
    #     print("-" * 40+"Key、Value"+"-" * 40)  # 添加分隔线
    """

    """ 调试打印：imdb_dict['dob']
    # 调试打印：imdb_dict['dob']
    # for dob in imdb_dict['dob']:
    #     print("-" * 40+"dob"+"-" * 40)  # 添加分隔线
    #     print(dob)      #719205
    #     temp=reformat_date(dob)
    #     print("-" * 40+"temp"+"-" * 40)  # 添加分隔线
    #     print(temp)     #年份：1969
    """
    imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]

    """ 调试打印：imdb_dict['full_path']
    # 调试打印：imdb_dict['dob']
    # for path in imdb_dict['full_path']:
    #     print("-" * 40+" full_path "+"-" * 40)  # 添加分隔线
    #     print(path)      #['98/nm0000098_rm432248832_1969-2-11_2000.jpg']
    #     temp=create_path(path)
    #     print("-" * 40+" path "+"-" * 40)  # 添加分隔线
    #     print(temp)     #path：data/imdb_crop/98/nm0000098_rm432248832_1969-2-11_2000.jpg
    """
    imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

    # 向字典中添加‘age’键  计算一个人的年龄
    imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

    print("字典创建...")

    raw_path = imdb_dict['full_path']
    raw_age = imdb_dict['age']
    raw_gender = imdb_dict['gender']
    raw_sface = imdb_dict['second_face_score']
    raw_face_loc = imdb_dict['face_location']

    # age = []类型：Python 原生列表（list）。创建一个空的 Python 列表。
    # 适用于存储不同类型的对象（可以是任意类型的元素），并支持动态大小调整。
    
    # samples = [100]，类型：Python 原生列表（list）。
    # 创建一个包含一个元素 100 的列表。

    # current_age = np.zeros(101)类型：NumPy 数组（numpy.ndarray）
    # 创建一个包含 101 个元素的数组，所有元素的初始值为零。
    # NumPy 数组专门用于高效存储数值数据，并支持数学计算。
    age = []
    gender = []
    imgs = []
    samples = []
    current_age = np.zeros(101)
    temp=0  #是不是人脸
    # tqdm 是一个用于显示进度条的 Python 库，它能帮助你在执行长时间的循环或任务时查看进度，从而提高用户体验。
    # range(len(raw_sface))生成一个从 0 到 len(raw_sface) - 1 的整数序列，通常用于循环中表示索引。
    # for i in tqdm(range(len(raw_sface))):
    for i in tqdm(range(1000)): #测试数据1000
        sface = raw_sface[i]

        # np.isnan(sface):检查 sface 是否是一个 NaN（Not a Number）
        # raw_age[i] >= 0 and raw_age[i] <= 100：这个条件判断 raw_age[i] 的值是否在 0 到 100 之间，表示年龄必须在有效范围内
        # not np.isnan(raw_gender[i])：这个条件检查 raw_gender[i] 是否不是 NaN。
        if np.isnan(sface) and raw_age[i] >= 0 and raw_age[i] <= 100 and not np.isnan(raw_gender[i]):
            is_valid, face_location, landmarks = get_face_attributes(raw_path[i])

            #是有效的人脸
            if is_valid:

                # 定义一个临时变量 age_tmp，虽然在代码中未使用，但可能是为了后续的扩展或调试
                age_tmp = 0

                # 检查当前年龄（current_age[raw_age[i]])是否大于等于5000，如果是，则跳过
                if current_age[raw_age[i]] >= 5000:
                    continue
                # 如果年龄小于5000且有效，则将该年龄值添加到 age 列表中
                age.append(raw_age[i])

                # 将性别值添加到 gender 列表中
                gender.append(raw_gender[i])

                # 将图片路径（raw_path[i]）添加到 imgs 列表中
                imgs.append(raw_path[i])

                # 将包含年龄、性别、图片路径、人脸位置和关键点信息的数据保存到 samples 列表中
                samples.append({'age': int(raw_age[i]), 'gender': int(raw_gender[i]), 'full_path': raw_path[i],
                                'face_location': face_location, 'landmarks': landmarks})
                # print("-" * 40+"  samples:"+ str(i) +"-" * 40)  # 添加分隔线
                # print(samples)
                # 更新 current_age 字典中当前年龄 raw_age[i] 的计数，表示该年龄已经被处理过一次
                current_age[raw_age[i]] += 1
            else:
                temp+=1
                print("-" * 40+"  不是人脸总个数:"+str(temp) +"-" * 40)  # 添加分隔线

    try:
        # 将 samples 列表中的元素随机打乱
        np.random.shuffle(samples)

        # 以二进制写入模式打开一个文件，用 pickle_file 指定文件路径
        f = open(pickle_file, 'wb')

        # 创建一个字典保存需要保存的数据
        save = {
            'age': age,     # 将 age 列表保存到字典的 'age' 键中
            'gender': gender,       # 将 gender 列表保存到字典的 'gender' 键中
            'samples': samples      # 将 samples 列表保存到字典的 'samples' 键中
        }

        # 使用 pickle 序列化字典 save，并写入文件 f 中
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

        # 关闭文件
        f.close()
    except Exception as e:

        # 打印错误信息
        print('无法将数据保存到', pickle_file, ':', e)

        # 抛出异常，终止程序
        raise
