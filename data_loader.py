import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 定义手指名称到数字标签的映射 (用于任务3)
# 0-4: 左手 thumb, index, middle, ring, little
# 5-9: 右手 thumb, index, middle, ring, little
FINGER_MAP = {
    "Left_thumb": 0, "Left_index": 1, "Left_middle": 2, "Left_ring": 3, "Left_little": 4,
    "Right_thumb": 5, "Right_index": 6, "Right_middle": 7, "Right_ring": 8, "Right_little": 9
}

# 定义性别到数字标签的映射 (用于任务4)
GENDER_MAP = {"M": 0, "F": 1}

class FingerprintDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = [] # Store all relevant labels

        # 正则表达式匹配文件名
        # 示例: 101__M_Right_middle_finger.BMP
        pattern = re.compile(r"(\d+)__([MF])_([LR]ight)_([a-z]+)_finger\.BMP")

        print(f"Scanning directory: {self.data_dir}")
        for filename in os.listdir(self.data_dir):
            match = pattern.match(filename)
            if match:
                subject_id, gender, hand, finger_name = match.groups()
                finger_full_name = f"{hand}_{finger_name}"

                if finger_full_name not in FINGER_MAP:
                    print(f"Warning: Skipping file with unknown finger name: {filename}")
                    continue

                img_path = os.path.join(self.data_dir, filename)
                self.image_files.append(img_path)
                self.labels.append({
                    "subject_id": int(subject_id),
                    "gender_str": gender,
                    "gender_label": GENDER_MAP[gender],
                    "finger_str": finger_full_name,
                    "finger_label": FINGER_MAP[finger_full_name],
                    "path": img_path
                })
        print(f"Found {len(self.image_files)} images.")
        if not self.image_files:
             raise FileNotFoundError(f"No valid image files found in {self.data_dir}. Check the directory and file naming.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # 使用 OpenCV 读取灰度图
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read image: {img_path}")

        # 获取该图像的所有标签
        labels = self.labels[idx]

        if self.transform:
            # torchvision 的 transform 通常期望 PIL Image 或 HWC numpy array
            # ToTensor 会自动将 HWC [0, 255] -> CHW [0.0, 1.0]
            # 对于灰度图，它会变成 [1, H, W]
            image = self.transform(image)
        else:
             # 如果没有 transform，手动转换为 tensor 并添加通道维度
             image = torch.from_numpy(image).unsqueeze(0).float() / 255.0


        # 返回图像和包含所有标签的字典
        return image, labels

class SubsetWithLabels(Subset):
    """扩展Subset类以支持直接访问标签属性"""
    def __getitem__(self, idx):
        image, label_dict = self.dataset[self.indices[idx]]
        return image, label_dict

def get_data_loaders(data_dir, batch_size=256, test_size=0.15, val_size=0.15, img_size=96, random_state=42):
    """
    加载数据，按 Subject ID 划分数据集，并创建 DataLoader。
    """
    # 定义图像转换
    train_transform = transforms.Compose([
        transforms.ToPILImage(), # cv2 读取的是 numpy array, 先转 PIL
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(), # 会自动将灰度 PIL (H, W) -> Tensor (1, H, W) 并归一化到 [0, 1]
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    full_dataset = FingerprintDataset(data_dir, transform=None)  # 先创建不带转换的数据集

    # 按 Subject ID 划分
    subject_ids = list(set(label['subject_id'] for label in full_dataset.labels))
    train_val_ids, test_ids = train_test_split(subject_ids, test_size=test_size, random_state=random_state)

    # 从 train_val_ids 中划分出 validation set
    # 注意 val_size 是相对于 *原始* 数据集的比例，需要调整
    relative_val_size = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=relative_val_size, random_state=random_state)

    # 创建索引映射
    train_indices = [i for i, label in enumerate(full_dataset.labels) if label['subject_id'] in train_ids]
    val_indices = [i for i, label in enumerate(full_dataset.labels) if label['subject_id'] in val_ids]
    test_indices = [i for i, label in enumerate(full_dataset.labels) if label['subject_id'] in test_ids]

    print(f"Total subjects: {len(subject_ids)}")
    print(f"Train subjects: {len(train_ids)} ({len(train_indices)} images)")
    print(f"Validation subjects: {len(val_ids)} ({len(val_indices)} images)")
    print(f"Test subjects: {len(test_ids)} ({len(test_indices)} images)")
    
    # 创建不同变换版本的数据集
    full_dataset_train = FingerprintDataset(data_dir, transform=train_transform)
    full_dataset_val = FingerprintDataset(data_dir, transform=test_transform)
    full_dataset_test = FingerprintDataset(data_dir, transform=test_transform)

    # 创建自定义Subset
    train_dataset = SubsetWithLabels(full_dataset_train, train_indices)
    val_dataset = SubsetWithLabels(full_dataset_val, val_indices)
    test_dataset = SubsetWithLabels(full_dataset_test, test_indices)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # 测试数据加载器
    DATA_DIR = 'dataset/SOCOFing_Real/'
    try:
        train_loader, val_loader, test_loader = get_data_loaders(DATA_DIR, batch_size=4, img_size=96)

        print("\nTesting Train Loader...")
        for i, (images, labels_dict) in enumerate(train_loader):
            print(f"Batch {i+1}:")
            print("Images shape:", images.shape) # 应该类似 [4, 1, 96, 96]
            print("Subject IDs:", [lbl['subject_id'] for lbl in labels_dict])
            print("Gender labels:", [lbl['gender_label'] for lbl in labels_dict])
            print("Finger labels:", [lbl['finger_label'] for lbl in labels_dict])
            if i == 0: # 只显示第一个 batch 的信息
                break

        print("\nTesting Validation Loader...")
        # ... 类似的测试代码 ...

        print("\nTesting Test Loader...")
        # ... 类似的测试代码 ...

        print("\nDataLoaders created successfully!")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() 