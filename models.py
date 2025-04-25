import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FingerCNN(nn.Module):
    """
    用于手指识别的CNN模型（任务3）
    输入: [B, 1, H, W] 的灰度指纹图像
    输出: [B, 10] 的logits，对应10个类别（左右手的5个手指）
    """
    def __init__(self, img_size=96):
        super(FingerCNN, self).__init__()
        self.img_size = img_size
        
        # 卷积块1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积块4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算卷积后的特征图大小
        feature_size = img_size // 16  # 四次2x2池化后的尺寸 (96/16=6)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * feature_size * feature_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # 10个类别（左右手各5个手指）
        
    def forward(self, x):
        # 卷积块1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 卷积块4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

class GenderCNN(nn.Module):
    """
    用于性别识别的CNN模型（任务4）
    输入: [B, 1, H, W] 的灰度指纹图像
    输出: [B, 1] 的logits，表示性别的概率（sigmoid后0表示男性，1表示女性）
    """
    def __init__(self, img_size=96):
        super(GenderCNN, self).__init__()
        self.img_size = img_size
        
        # 这里我们使用更轻量的网络结构，因为性别分类可能比手指分类更简单
        # 卷积块1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算卷积后的特征图大小
        feature_size = img_size // 8  # 三次2x2池化后的尺寸 (96/8=12)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * feature_size * feature_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # 二分类（男/女）
        
    def forward(self, x):
        # 卷积块1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

class SiameseNetwork(nn.Module):
    """
    用于指纹验证的孪生网络（任务2）
    输入: 两张指纹图像 [B, 1, H, W]
    输出: 相似度分数 [B, 1]，表示两张指纹来自同一个人的概率
    """
    def __init__(self, img_size=96):
        super(SiameseNetwork, self).__init__()
        
        # 特征提取器 - 使用较深的CNN来提取有效特征
        self.feature_extractor = nn.Sequential(
            # 卷积块1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化到 1x1
        )
        
        # 特征维度 = 512
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 512),  # 连接两个特征向量
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward_one(self, x):
        """前向传播单个图像以获取特征向量"""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # 展平为 [B, 512]
        return x
    
    def forward(self, x1, x2):
        """前向传播一对图像，计算相似度"""
        # 获取两张图像的特征向量
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 连接特征向量
        combined = torch.cat((feat1, feat2), 1)
        
        # 通过FC层计算相似度分数
        similarity = self.fc(combined)
        
        return similarity

# 使用迁移学习（如果需要更强大的模型）
class ResNet18Fingerprint(nn.Module):
    """
    使用预训练ResNet18作为特征提取器的手指识别模型（任务3的替代模型）
    """
    def __init__(self, num_classes=10):
        super(ResNet18Fingerprint, self).__init__()
        
        # 加载预训练的ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 修改第一个卷积层以接受单通道灰度图
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改最后的全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 为任务1添加的条件生成器网络（草稿，会在后续实现）
class FingerprintGenerator(nn.Module):
    """
    用于根据9个指纹生成第10个指纹的网络（任务1）
    这是一个草稿模型，需要进一步完善
    """
    def __init__(self):
        super(FingerprintGenerator, self).__init__()
        # TODO: 实现生成网络
        pass
    
    def forward(self, x):
        # TODO: 实现前向传播
        pass

# 简单测试代码
if __name__ == "__main__":
    # 测试FingerCNN
    model = FingerCNN(img_size=96)
    dummy_input = torch.randn(4, 1, 96, 96)  # 批量大小为4，单通道，96x96图像
    output = model(dummy_input)
    print(f"FingerCNN output shape: {output.shape}")  # 应该是 [4, 10]
    
    # 测试GenderCNN
    model = GenderCNN(img_size=96)
    dummy_input = torch.randn(4, 1, 96, 96)
    output = model(dummy_input)
    print(f"GenderCNN output shape: {output.shape}")  # 应该是 [4, 1]
    
    # 测试SiameseNetwork
    model = SiameseNetwork(img_size=96)
    dummy_input1 = torch.randn(4, 1, 96, 96)
    dummy_input2 = torch.randn(4, 1, 96, 96)
    output = model(dummy_input1, dummy_input2)
    print(f"SiameseNetwork output shape: {output.shape}")  # 应该是 [4, 1] 