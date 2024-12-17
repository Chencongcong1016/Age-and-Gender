import json
import pickle
import random

import cv2 as cv
import numpy as np

from config import *
from utils import align_face

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    num_samples = len(samples)
    num_train = int(train_split * num_samples)
    samples = samples[num_train:]

    samples = random.sample(samples, 10)

    inputs = torch.zeros([10, 3, image_h, image_w], dtype=torch.float, device=device)

    sample_preds = []

    for i, sample in enumerate(samples):
        full_path = sample['full_path']
        landmarks = sample['landmarks']
        age = sample['age']
        gender = sample['gender']
        print(full_path)
        raw = cv.imread(full_path)
        raw = cv.resize(raw, (image_w, image_h))
        filename = 'images/{}_raw.jpg'.format(i)
        cv.imwrite(filename, raw)
        img = align_face(full_path, landmarks)
        filename = 'images/{}_img.jpg'.format(i)
        cv.imwrite(filename, img)
        # 1. 打印 img 的形状，确认它的维度
        print(f"Original shape of img: {img.shape}")

        # 2. 根据 img 的形状，调整 transpose 操作
        # 如果 img 的形状是 (H, W, C)，那么我们可以使用 transpose(2, 0, 1)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            # 3D 数组 (H, W, C) -> (C, H, W)
            img = img.transpose(2, 0, 1)
            print(f"New shape of img after transpose (2, 0, 1): {img.shape}")
        else:
            print("The shape of img is not compatible with (H, W, C) format.")

        # 3. 处理其他情况
        # 如果 img 的形状不是 (H, W, C)，需要处理其他格式

        # 假设 img 的形状为 (C, H, W)，那么我们使用 transpose(0, 2, 1)
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(0, 2, 1)
            print(f"New shape of img after transpose (0, 2, 1): {img.shape}")

        # 如果 img 的形状是 (H, C, W)，我们使用 transpose(1, 0, 2)
        if len(img.shape) == 3 and img.shape[1] == 3:
            img = img.transpose(1, 0, 2)
            print(f"New shape of img after transpose (1, 0, 2): {img.shape}")
        # img = img.transpose(2, 0, 1)
        # assert img.shape == (3, image_h, image_w)
        # assert np.max(img) <= 255
        inputs[i] = torch.FloatTensor(img / 255.)

        sample_preds.append({'i': i, 'gen_true': gender, 'age_true': age, })

    checkpoint = 'BEST_checkpoint_.pth.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        age_out, gen_out = model(inputs)

    _, age_out = age_out.topk(1, 1, True, True)
    _, gen_out = gen_out.topk(1, 1, True, True)

    age_out = age_out.cpu().numpy()
    gen_out = gen_out.cpu().numpy()

    for i in range(10):
        sample = sample_preds[i]

        sample['gen_out'] = int(gen_out[i][0])
        sample['age_out'] = int(age_out[i][0])
        # print(gen_out[i][0])
        # print(type(gen_out[i][0]))

    # print('age_out.shape: ' + str(age_out.shape))
    # print('gen_out.shape: ' + str(gen_out.shape))
    # print('age_out.: ' + str(age_out))
    # print('gen_out.: ' + str(gen_out))

    with open('sample_preds.json', 'w') as file:
        json.dump(sample_preds, file, indent=4, ensure_ascii=False)
