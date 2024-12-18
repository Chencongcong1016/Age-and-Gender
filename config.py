import torch

image_w = 112
image_h = 112
channel = 3
epochs = 10000
patience = 10

# 模型参数
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置模型和PyTorch张量的设备
train_split = 0.9
age_num_classes = 101
gen_num_classes = 2

# 训练参数
start_epoch = 0
epochs = 120  # 训练的周期数（如果未触发提前停止）
epochs_since_improvement = 0  # 跟踪epoch的数量，因为在验证BLEU中有了改进
batch_size = 32
workers = 1  # 数据加载;现在，只有1与h5py一起工作
lr = 1e-4  # 学习速率
grad_clip = 5.  # 的绝对值剪辑梯度
print_freq = 100  # 每__批打印培训/验证统计数据
checkpoint = None  # 到检查点的路径，如果没有就没有

# 数据参数
DATA_DIR = 'data'
IMG_DIR = 'data/imdb_crop'
pickle_file = DATA_DIR + '/' + 'imdb-gender-age101.pkl'
