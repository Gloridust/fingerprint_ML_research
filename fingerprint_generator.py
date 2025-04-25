import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2
from torch.amp import autocast, GradScaler  # 更新混合精度训练导入

from data_loader import get_data_loaders
from models import FingerCNN  # 我们将使用部分预训练的手指识别模型特征

# 定义一个Encoder-Decoder结构，利用9个指纹的特征来生成第10个
class FingerprintEncoderDecoder(nn.Module):
    def __init__(self, img_size=96, feature_dim=64, latent_dim=128, target_finger_idx=0):
        """
        指纹生成模型
        
        Args:
            img_size: 输入图像的大小
            feature_dim: 卷积特征维度
            latent_dim: 潜在空间维度
            target_finger_idx: 要生成的目标手指索引(0-9)
        """
        super(FingerprintEncoderDecoder, self).__init__()
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.target_finger_idx = target_finger_idx
        
        # Encoder - 处理9个指纹图像，每个图像单独编码
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 48x48
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 24x24
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 12x12
            
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 6x6
        )
        
        # 每个指纹的特征向量大小
        self.feature_size = feature_dim * (img_size // 16) * (img_size // 16)
        
        # 融合9个特征向量的全连接层
        self.fusion = nn.Sequential(
            nn.Linear(9 * self.feature_size, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # 条件嵌入 - 将目标手指索引嵌入为向量并与融合特征连接
        self.finger_embedding = nn.Embedding(10, latent_dim // 4)
        
        # Decoder - 从潜在向量重构目标指纹
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 4, 6 * 6 * 128),
            nn.BatchNorm1d(6 * 6 * 128),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 48x48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # -> 96x96
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def encode(self, x):
        """
        编码9个指纹图像
        
        Args:
            x: 输入张量，形状 [batch_size, 9, 1, img_size, img_size]
        
        Returns:
            latent: 融合后的潜在向量，形状 [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        features = []
        
        # 单独编码每个指纹
        for i in range(9):
            # 从第i个指纹获取特征
            feat = self.encoder_conv(x[:, i])
            feat = feat.view(batch_size, -1)
            features.append(feat)
        
        # 连接所有特征
        features = torch.cat(features, dim=1)
        
        # 融合特征
        latent = self.fusion(features)
        
        return latent
    
    def decode(self, latent, finger_idx):
        """
        解码潜在向量为目标指纹图像
        
        Args:
            latent: 融合后的潜在向量，形状 [batch_size, latent_dim]
            finger_idx: 目标手指索引，形状 [batch_size]
        
        Returns:
            reconstructed: 重构的指纹图像，形状 [batch_size, 1, img_size, img_size]
        """
        # 获取手指嵌入
        finger_emb = self.finger_embedding(finger_idx)
        
        # 连接潜在向量和手指嵌入
        x = torch.cat([latent, finger_emb], dim=1)
        
        # 解码
        x = self.decoder_fc(x)
        x = x.view(-1, 128, 6, 6)
        reconstructed = self.decoder_conv(x)
        
        return reconstructed
    
    def forward(self, x, finger_idx):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 [batch_size, 9, 1, img_size, img_size]
            finger_idx: 目标手指索引，形状 [batch_size]
        
        Returns:
            reconstructed: 重构的指纹图像，形状 [batch_size, 1, img_size, img_size]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent, finger_idx)
        return reconstructed

class FingerprintGeneratorDataset(Dataset):
    """
    为指纹生成任务准备数据的数据集类
    """
    def __init__(self, dataset, target_finger_idx=None):
        """
        Args:
            dataset: 原始指纹数据集
            target_finger_idx: 目标手指索引。如果为None，则随机选择
        """
        self.dataset = dataset
        self.target_finger_idx = target_finger_idx
        
        # 按subject_id分组图像
        self.subject_data = {}
        
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            subject_id = label['subject_id']
            finger_idx = label['finger_label']
            
            if subject_id not in self.subject_data:
                self.subject_data[subject_id] = {}
            
            # 添加图像及其标签
            self.subject_data[subject_id][finger_idx] = (img, label)
        
        # 只保留同时有10个手指数据的subject
        self.valid_subjects = []
        for subject_id, fingers in self.subject_data.items():
            if len(fingers) == 10:  # 有全部10个手指的数据
                self.valid_subjects.append(subject_id)
        
        print(f"有效subjects数量: {len(self.valid_subjects)}")
        if len(self.valid_subjects) == 0:
            raise ValueError("没有找到拥有全部10个手指数据的subjects")
    
    def __len__(self):
        return len(self.valid_subjects)
    
    def __getitem__(self, idx):
        subject_id = self.valid_subjects[idx]
        finger_data = self.subject_data[subject_id]
        
        # 如果目标手指索引未指定，则随机选择一个
        if self.target_finger_idx is None:
            target_idx = np.random.randint(0, 10)  # 随机选择一个手指作为目标
        else:
            target_idx = self.target_finger_idx
        
        # 目标指纹
        target_img, target_label = finger_data[target_idx]
        
        # 输入指纹列表 (除了目标指纹外的9个)
        input_imgs = []
        for i in range(10):
            if i != target_idx:
                input_imgs.append(finger_data[i][0])
        
        # 将输入图像堆叠成一个张量
        input_tensor = torch.stack(input_imgs)
        
        return {
            'input_imgs': input_tensor,  # 形状 [9, 1, img_size, img_size]
            'target_img': target_img,    # 形状 [1, img_size, img_size]
            'target_idx': target_idx,    # 目标手指的索引
            'subject_id': subject_id     # subject ID
        }

def train_fingerprint_generator(data_dir, batch_size=8, epochs=30, learning_rate=0.0005, 
                               img_size=96, model_save_path='models/fingerprint_generator.pth',
                               target_finger=None, patience=5):
    """
    训练指纹生成模型
    
    Args:
        data_dir: 数据集路径
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        img_size: 图像大小
        model_save_path: 模型保存路径
        target_finger: 特定目标手指索引。如果为None，则随机生成不同手指
        patience: 早停机制的耐心参数
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, batch_size=batch_size, img_size=img_size
    )
    
    # 3. 创建生成器数据集
    print("准备训练集...")
    train_gen_dataset = FingerprintGeneratorDataset(train_loader.dataset, target_finger_idx=target_finger)
    train_gen_loader = DataLoader(train_gen_dataset, batch_size=batch_size, shuffle=True)
    
    print("准备验证集...")
    val_gen_dataset = FingerprintGeneratorDataset(val_loader.dataset, target_finger_idx=target_finger)
    val_gen_loader = DataLoader(val_gen_dataset, batch_size=batch_size, shuffle=False)
    
    print("准备测试集...")
    test_gen_dataset = FingerprintGeneratorDataset(test_loader.dataset, target_finger_idx=target_finger)
    test_gen_loader = DataLoader(test_gen_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. 创建模型
    model = FingerprintEncoderDecoder(img_size=img_size, target_finger_idx=target_finger)
    model = model.to(device)
    
    # 5. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 添加混合精度训练的scaler
    scaler = GradScaler()
    
    # 6. 训练模型
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 早停相关变量
    early_stopping_counter = 0
    
    print(f"开始训练，共{epochs}轮...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        
        for batch in train_gen_loader:
            input_imgs = batch['input_imgs'].to(device)
            target_img = batch['target_img'].to(device)
            target_idx = batch['target_idx'].to(device)
            
            # 前向传播 - 使用混合精度
            with autocast(device_type='cuda'):
                output = model(input_imgs, target_idx)
                loss = criterion(output, target_img)
            
            # 反向传播和优化 - 使用混合精度
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
        
        # 计算平均损失
        epoch_train_loss /= len(train_gen_loader)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch in val_gen_loader:
                input_imgs = batch['input_imgs'].to(device)
                target_img = batch['target_img'].to(device)
                target_idx = batch['target_idx'].to(device)
                
                # 前向传播 - 使用混合精度
                with autocast(device_type='cuda'):
                    output = model(input_imgs, target_idx)
                    loss = criterion(output, target_img)
                
                epoch_val_loss += loss.item()
        
        # 计算平均损失
        epoch_val_loss /= len(val_gen_loader)
        val_losses.append(epoch_val_loss)
        
        # 调整学习率
        scheduler.step(epoch_val_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f}")
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            # 如果验证损失没有改善，增加计数器
            early_stopping_counter += 1
            print(f"早停计数器: {early_stopping_counter}/{patience}")
        
        # 检查早停条件
        if early_stopping_counter >= patience:
            print(f"早停机制触发，停止训练。当前轮次: {epoch+1}")
            break
    
    # 7. 在测试集上评估最佳模型
    print("\n在测试集上评估...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    test_loss = 0
    test_ssim = 0
    num_samples = 0
    
    # 用于可视化的样本
    vis_samples = []
    
    with torch.no_grad():
        for batch in test_gen_loader:
            input_imgs = batch['input_imgs'].to(device)
            target_img = batch['target_img'].to(device)
            target_idx = batch['target_idx'].to(device)
            
            # 前向传播
            output = model(input_imgs, target_idx)
            
            # 计算MSE损失
            loss = criterion(output, target_img)
            test_loss += loss.item() * target_img.size(0)
            
            # 计算SSIM
            for i in range(target_img.size(0)):
                true_img = target_img[i, 0].cpu().numpy()
                pred_img = output[i, 0].cpu().numpy()
                ssim_val = ssim(true_img, pred_img, data_range=1.0)
                test_ssim += ssim_val
            
            num_samples += target_img.size(0)
            
            # 保存一些样本用于可视化
            if len(vis_samples) < 5:
                for i in range(min(3, target_img.size(0))):
                    sample = {
                        'input_imgs': input_imgs[i].cpu(),
                        'target_img': target_img[i].cpu(),
                        'output': output[i].cpu(),
                        'target_idx': target_idx[i].item()
                    }
                    vis_samples.append(sample)
    
    # 计算平均指标
    test_loss /= num_samples
    test_ssim /= num_samples
    
    print(f"测试集平均MSE: {test_loss:.6f}")
    print(f"测试集平均SSIM: {test_ssim:.6f}")
    
    # 8. 可视化结果
    print("\n生成可视化结果...")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('MSE损失')
    plt.legend()
    plt.title('训练和验证损失')
    plt.savefig('results/fingerprint_generator_loss.png')
    plt.show()
    
    # 可视化生成的指纹
    plt.figure(figsize=(15, len(vis_samples) * 3))
    
    finger_names = ['Left_thumb', 'Left_index', 'Left_middle', 'Left_ring', 'Left_little',
                   'Right_thumb', 'Right_index', 'Right_middle', 'Right_ring', 'Right_little']
    
    for i, sample in enumerate(vis_samples):
        # 显示目标图像
        plt.subplot(len(vis_samples), 3, i*3 + 1)
        plt.imshow(sample['target_img'][0], cmap='gray')
        plt.title(f'真实指纹 ({finger_names[sample["target_idx"]]})')
        plt.axis('off')
        
        # 显示生成的图像
        plt.subplot(len(vis_samples), 3, i*3 + 2)
        plt.imshow(sample['output'][0], cmap='gray')
        plt.title(f'生成的指纹')
        plt.axis('off')
        
        # 显示输入图像的一部分
        plt.subplot(len(vis_samples), 3, i*3 + 3)
        plt.imshow(sample['input_imgs'][0][0], cmap='gray')
        plt.title(f'输入指纹示例')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/fingerprint_generator_samples.png')
    plt.show()
    
    return model, (train_losses, val_losses)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练指纹生成模型')
    parser.add_argument('--data_dir', type=str, default='dataset/SOCOFing_Real/', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--img_size', type=int, default=96, help='图像大小')
    parser.add_argument('--target_finger', type=int, default=None, help='目标手指索引 (0-9)，为None则随机')
    parser.add_argument('--patience', type=int, default=5, help='早停机制的耐心参数')
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    
    train_fingerprint_generator(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        img_size=args.img_size,
        target_finger=args.target_finger,
        patience=args.patience
    ) 