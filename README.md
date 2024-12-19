# 同时识别年龄与性别
基于PyTorch 实现多任务学习，在同时识别年龄与性别。


## 数据集

[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) 数据集，460,723张图片。

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/imdb-wiki-teaser.png)

这里为简洁只使用 IMDB 数据集。

### 数据类型
    dob：出生日期 （Matlab serial date number）
    photo_taken：照片拍摄年份
    full_path：文件路径
    性别：女性 0 个，男性 1 个，未知时为 NaN
    name：名人姓名
    face_location：面部的位置。要在 Matlab 中裁剪脸部，请运行
    IMG（face_location（2）：face_location（4），face_location（1）：face_location（3），:)）
    face_score：detector 分数（越高越好）。Inf表示在图像中没有找到人脸，然后 face_location只返回整个图像
    second_face_score：得分第二高的人脸的 detector 分数。这对于忽略具有多个面的图像非常有用。如果未检测到第二张人脸，则 second_face_score为 NaN。
    celeb_names （仅限 IMDB）：所有名人姓名列表
    celeb_id（仅限 IMDB）：名人姓名索引

### 年龄分布：

460723张照片为平衡每个年龄最多只保存5000张，清洗后得到163065张，按年龄分布作图：

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/age.png)

## 安装
安装：pip install scipy

卸载：pip uninstall scipy

## 用法

### 数据预处理
提取163065张图片：
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

### 准确度比较

#|图片大小|网络|性别准确度(验证集)|年龄MAE(验证集)|年龄损失函数|批量大小|Loss|年龄权重|
|---|---|---|---|---|---|---|---|---|
|1|112x112|ResNet-18|90.756%|5.710|回归(L1Loss)|32|0.9757|0.1|
|2|224x224|ResNet-18|90.887%|5.694|回归(L1Loss)|32|0.9719|0.1|
|3|112x112|ResNet-18|90.140%|5.986|回归(L2Loss)|32|1.121|0.01|
|4|224x224|ResNet-18|90.064%|8.475|分类(交叉熵)|32|TBD|TBD|
|5|224x224|ResNet-50|90.034%|TBD|分类(交叉熵)|32|TBD|TBD|


### Demo
下载预训练的模型 [Link](https://github.com/foamliu/Age-and-Gender/releases/download/1.0/BEST_checkpoint_.pth.tar)，执行：
```bash
$ python demo.py
```

效果图如下：

原图 | 校准 | 识别 | 标注 |
|---|---|---|---|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/0_img.jpg)|性别：女, 年龄：29|性别：女, 年龄：24|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/1_img.jpg)|性别：女, 年龄：29|性别：男, 年龄：26|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/2_img.jpg)|性别：男, 年龄：34|性别：男, 年龄：49|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/3_img.jpg)|性别：女, 年龄：29|性别：女, 年龄：29|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/4_img.jpg)|性别：男, 年龄：36|性别：女, 年龄：23|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/5_img.jpg)|性别：男, 年龄：29|性别：男, 年龄：15|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/6_img.jpg)|性别：男, 年龄：34|性别：男, 年龄：32|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/7_img.jpg)|性别：男, 年龄：42|性别：男, 年龄：42|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/8_img.jpg)|性别：男, 年龄：36|性别：女, 年龄：31|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/9_img.jpg)|性别：男, 年龄：39|性别：男, 年龄：59|

