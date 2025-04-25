import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.amp import autocast, GradScaler  # 更新混合精度训练导入

from data_loader import get_data_loaders
from models import FingerCNN, GenderCNN, SiameseNetwork, ResNet18Fingerprint

# 设置随机种子以确保可重现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False  # 改为False以提高速度
    torch.backends.cudnn.benchmark = True  # 启用基准测试以提高速度

def train_finger_classifier(data_dir, batch_size=256, epochs=15, learning_rate=0.001, 
                            img_size=96, model_save_path='models/finger_classifier.pth', 
                            use_resnet=False, patience=5):  # 添加耐心参数
    """
    训练用于手指分类的模型（任务3）
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, batch_size=batch_size, img_size=img_size
    )
    
    # 3. 创建模型
    if use_resnet:
        print("Using ResNet18 model")
        model = ResNet18Fingerprint(num_classes=10)
    else:
        print("Using custom FingerCNN model")
        model = FingerCNN(img_size=img_size)
    
    model = model.to(device)
    
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 添加混合精度训练的scaler
    scaler = GradScaler()
    
    # 5. 训练模型
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    
    # 早停相关变量
    early_stopping_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    
    # 添加训练速度监控
    start_time = time.time()
    batch_times = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        train_preds = []
        train_targets = []
        epoch_start = time.time()
        
        for i, (images, labels_dict) in enumerate(train_loader):
            batch_start = time.time()
            
            images = images.to(device)
            # 处理标签 - 修复了这里的bug
            # 检查labels_dict的类型并相应地处理
            if isinstance(labels_dict, list):
                finger_labels = torch.tensor([label['finger_label'] for label in labels_dict], dtype=torch.long).to(device)
            else:
                # 如果是Subset数据集,则可能直接返回tensor
                finger_labels = labels_dict['finger_label'].to(device) if isinstance(labels_dict, dict) else labels_dict.to(device)
            
            # 前向传播 - 使用混合精度
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, finger_labels)
            
            # 反向传播和优化 - 使用混合精度
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失和预测结果
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(finger_labels.cpu().numpy())
            
            # 记录批次时间
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # 每10个批次打印一次进度和GPU使用情况
            if (i + 1) % 10 == 0:
                images_per_sec = images.shape[0] / batch_time
                print(f"  Batch {i+1}/{len(train_loader)}: {images_per_sec:.2f} imgs/sec, batch time: {batch_time:.3f}s")
                if torch.cuda.is_available():
                    print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        
        # 计算平均损失和准确率
        epoch_train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        
        epoch_time = time.time() - epoch_start
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels_dict in val_loader:
                images = images.to(device)
                # 处理标签 - 修复了这里的bug
                if isinstance(labels_dict, list):
                    finger_labels = torch.tensor([label['finger_label'] for label in labels_dict], dtype=torch.long).to(device)
                else:
                    # 如果是Subset数据集,则可能直接返回tensor
                    finger_labels = labels_dict['finger_label'].to(device) if isinstance(labels_dict, dict) else labels_dict.to(device)
                
                # 前向传播 - 也使用混合精度
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, finger_labels)
                
                # 记录损失和预测结果
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(finger_labels.cpu().numpy())
        
        # 计算平均损失和准确率
        epoch_val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)
        
        # 调整学习率
        scheduler.step(epoch_val_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # 确保保存路径存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            # 如果验证损失没有改善，增加计数器
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{patience}")
        
        # 检查早停条件
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 6. 在测试集上评估最佳模型
    print("\nEvaluating on test set...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels_dict in test_loader:
            images = images.to(device)
            # 处理标签 - 修复了这里的bug
            if isinstance(labels_dict, list):
                finger_labels = torch.tensor([label['finger_label'] for label in labels_dict], dtype=torch.long).to(device)
            else:
                # 如果是Subset数据集,则可能直接返回tensor
                finger_labels = labels_dict['finger_label'].to(device) if isinstance(labels_dict, dict) else labels_dict.to(device)
            
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(finger_labels.cpu().numpy())
    
    # 计算评估指标
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds, average='macro')
    test_recall = recall_score(test_targets, test_preds, average='macro')
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # 7. 可视化结果
    print("\nGenerating visualizations...")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('results/finger_classifier_training_curves.png')
    plt.show()
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_targets, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/finger_classifier_confusion_matrix.png')
    plt.show()
    
    return model, (train_losses, val_losses, train_accs, val_accs)

def train_gender_classifier(data_dir, batch_size=256, epochs=15, learning_rate=0.001, 
                            img_size=96, model_save_path='models/gender_classifier.pth',
                            patience=5):  # 添加耐心参数
    """
    训练用于性别分类的模型（任务4）
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, batch_size=batch_size, img_size=img_size
    )
    
    # 3. 创建模型
    model = GenderCNN(img_size=img_size)
    model = model.to(device)
    
    # 4. 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 带有sigmoid的二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 5. 训练模型
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    
    # 早停相关变量
    early_stopping_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        train_preds = []
        train_targets = []
        
        for images, labels_dict in train_loader:
            images = images.to(device)
            # 处理标签 - 修复了这里的bug
            if isinstance(labels_dict, list):
                gender_labels = torch.tensor([label['gender_label'] for label in labels_dict], dtype=torch.float).unsqueeze(1).to(device)
            else:
                # 如果是Subset数据集,则可能直接返回tensor或字典
                gender_labels = labels_dict['gender_label'].unsqueeze(1).to(device) if isinstance(labels_dict, dict) else labels_dict.unsqueeze(1).to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, gender_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失和预测结果
            epoch_train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(gender_labels.cpu().numpy())
        
        # 计算平均损失和准确率
        epoch_train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels_dict in val_loader:
                images = images.to(device)
                # 处理标签 - 修复了这里的bug
                if isinstance(labels_dict, list):
                    gender_labels = torch.tensor([label['gender_label'] for label in labels_dict], dtype=torch.float).unsqueeze(1).to(device)
                else:
                    # 如果是Subset数据集,则可能直接返回tensor或字典
                    gender_labels = labels_dict['gender_label'].unsqueeze(1).to(device) if isinstance(labels_dict, dict) else labels_dict.unsqueeze(1).to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, gender_labels)
                
                # 记录损失和预测结果
                epoch_val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(gender_labels.cpu().numpy())
        
        # 计算平均损失和准确率
        epoch_val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)
        
        # 调整学习率
        scheduler.step(epoch_val_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # 确保保存路径存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            # 如果验证损失没有改善，增加计数器
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{patience}")
        
        # 检查早停条件
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 6. 在测试集上评估最佳模型
    print("\nEvaluating on test set...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels_dict in test_loader:
            images = images.to(device)
            # 处理标签 - 修复了这里的bug
            if isinstance(labels_dict, list):
                gender_labels = torch.tensor([label['gender_label'] for label in labels_dict], dtype=torch.float).unsqueeze(1).to(device)
            else:
                # 如果是Subset数据集,则可能直接返回tensor或字典
                gender_labels = labels_dict['gender_label'].unsqueeze(1).to(device) if isinstance(labels_dict, dict) else labels_dict.unsqueeze(1).to(device)
            
            # 前向传播
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(gender_labels.cpu().numpy())
    
    # 计算评估指标
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds, average='binary')
    test_recall = recall_score(test_targets, test_preds, average='binary')
    test_f1 = f1_score(test_targets, test_preds, average='binary')
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # 7. 可视化结果
    print("\nGenerating visualizations...")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('results/gender_classifier_training_curves.png')
    plt.show()
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_targets, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/gender_classifier_confusion_matrix.png')
    plt.show()
    
    return model, (train_losses, val_losses, train_accs, val_accs)

def create_pairs_for_siamese(data_loader, num_pairs=1000, same_person_ratio=0.5):
    """
    为孪生网络创建图像对和标签
    """
    all_images = []
    all_subjects = []
    
    # 收集所有图像和对应的subject_id
    print("Collecting images for creating pairs...")
    for images, labels_dict in data_loader:
        for i in range(images.shape[0]):
            all_images.append(images[i])
            all_subjects.append(labels_dict[i]['subject_id'])
    
    all_images = torch.stack(all_images)
    all_subjects = np.array(all_subjects)
    
    # 创建图像对和标签
    print(f"Creating {num_pairs} pairs with {same_person_ratio*100:.0f}% same-person pairs...")
    
    pairs = []
    labels = []
    
    num_same = int(num_pairs * same_person_ratio)
    num_diff = num_pairs - num_same
    
    # 创建相同person的对
    unique_subjects = np.unique(all_subjects)
    
    # 确保每个人有足够的图像
    subject_counts = {s: np.sum(all_subjects == s) for s in unique_subjects}
    valid_subjects = [s for s, count in subject_counts.items() if count >= 2]
    
    if len(valid_subjects) == 0:
        raise ValueError("No subjects with at least 2 samples found in the dataset")
    
    same_count = 0
    attempts = 0
    max_attempts = num_same * 10
    
    while same_count < num_same and attempts < max_attempts:
        subject = np.random.choice(valid_subjects)
        
        # 找到该subject的所有图像索引
        indices = np.where(all_subjects == subject)[0]
        
        if len(indices) >= 2:
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            
            # 添加图像对和标签（1表示相同人）
            pairs.append((all_images[idx1], all_images[idx2]))
            labels.append(1)
            
            same_count += 1
        
        attempts += 1
    
    if same_count < num_same:
        print(f"Warning: Could only create {same_count} same-person pairs")
    
    # 创建不同person的对
    diff_count = 0
    attempts = 0
    max_attempts = num_diff * 10
    
    while diff_count < num_diff and attempts < max_attempts:
        # 随机选择两个不同的subject
        if len(unique_subjects) < 2:
            raise ValueError("Need at least 2 different subjects to create different-person pairs")
        
        subject1, subject2 = np.random.choice(unique_subjects, 2, replace=False)
        
        # 找到对应的图像索引
        indices1 = np.where(all_subjects == subject1)[0]
        indices2 = np.where(all_subjects == subject2)[0]
        
        if len(indices1) > 0 and len(indices2) > 0:
            idx1 = np.random.choice(indices1)
            idx2 = np.random.choice(indices2)
            
            # 添加图像对和标签（0表示不同人）
            pairs.append((all_images[idx1], all_images[idx2]))
            labels.append(0)
            
            diff_count += 1
        
        attempts += 1
    
    if diff_count < num_diff:
        print(f"Warning: Could only create {diff_count} different-person pairs")
    
    print(f"Created {same_count + diff_count} pairs in total")
    
    return pairs, labels

def train_siamese_network(data_dir, batch_size=256, epochs=15, learning_rate=0.001, 
                          img_size=96, model_save_path='models/siamese_network.pth',
                          num_train_pairs=2000, num_val_pairs=500, num_test_pairs=1000,
                          patience=5):  # 添加耐心参数
    """
    训练用于指纹验证的孪生网络（任务2）
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, batch_size=batch_size, img_size=img_size
    )
    
    # 3. 创建图像对
    print("Creating pairs for training...")
    train_pairs, train_labels = create_pairs_for_siamese(train_loader, num_pairs=num_train_pairs)
    print("Creating pairs for validation...")
    val_pairs, val_labels = create_pairs_for_siamese(val_loader, num_pairs=num_val_pairs)
    print("Creating pairs for testing...")
    test_pairs, test_labels = create_pairs_for_siamese(test_loader, num_pairs=num_test_pairs)
    
    # 4. 创建模型
    model = SiameseNetwork(img_size=img_size)
    model = model.to(device)
    
    # 5. 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 6. 训练模型
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    
    # 早停相关变量
    early_stopping_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    
    # 添加训练速度监控
    start_time = time.time()
    batch_times = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        train_preds = []
        
        # 打乱训练对的顺序
        indices = np.random.permutation(len(train_pairs))
        train_batch_size = 32  # 可以调整
        
        for i in range(0, len(indices), train_batch_size):
            batch_indices = indices[i:i+train_batch_size]
            
            # 准备批次数据
            batch_img1 = torch.stack([train_pairs[idx][0] for idx in batch_indices]).to(device)
            batch_img2 = torch.stack([train_pairs[idx][1] for idx in batch_indices]).to(device)
            batch_labels = torch.tensor([train_labels[idx] for idx in batch_indices], dtype=torch.float).unsqueeze(1).to(device)
            
            # 前向传播
            outputs = model(batch_img1, batch_img2)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失和预测结果
            epoch_train_loss += loss.item() * len(batch_indices)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend([(predicted[j].item(), batch_labels[j].item()) for j in range(len(predicted))])
            
            # 记录批次时间
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # 每10个批次打印一次进度和GPU使用情况
            if (i + 1) % 10 == 0:
                images_per_sec = (len(batch_indices) * 2) / batch_time
                print(f"  Batch {i+1}/{len(train_pairs)}: {images_per_sec:.2f} imgs/sec, batch time: {batch_time:.3f}s")
                if torch.cuda.is_available():
                    print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        
        # 计算平均损失和准确率
        epoch_train_loss /= len(train_pairs)
        train_acc = sum(1 for pred, label in train_preds if pred == label) / len(train_preds)
        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_preds = []
        
        with torch.no_grad():
            val_batch_size = 32  # 可以调整
            
            for i in range(0, len(val_pairs), val_batch_size):
                end_idx = min(i + val_batch_size, len(val_pairs))
                
                # 准备批次数据
                batch_img1 = torch.stack([val_pairs[idx][0] for idx in range(i, end_idx)]).to(device)
                batch_img2 = torch.stack([val_pairs[idx][1] for idx in range(i, end_idx)]).to(device)
                batch_labels = torch.tensor([val_labels[idx] for idx in range(i, end_idx)], dtype=torch.float).unsqueeze(1).to(device)
                
                # 前向传播
                outputs = model(batch_img1, batch_img2)
                loss = criterion(outputs, batch_labels)
                
                # 记录损失和预测结果
                epoch_val_loss += loss.item() * (end_idx - i)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend([(predicted[j].item(), batch_labels[j].item()) for j in range(len(predicted))])
        
        # 计算平均损失和准确率
        epoch_val_loss /= len(val_pairs)
        val_acc = sum(1 for pred, label in val_preds if pred == label) / len(val_preds)
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)
        
        # 调整学习率
        scheduler.step(epoch_val_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # 确保保存路径存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            # 如果验证损失没有改善，增加计数器
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{patience}")
        
        # 检查早停条件
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 7. 在测试集上评估最佳模型
    print("\nEvaluating on test set...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    test_preds = []
    test_targets = []
    all_scores = []
    
    with torch.no_grad():
        test_batch_size = 32  # 可以调整
        
        for i in range(0, len(test_pairs), test_batch_size):
            end_idx = min(i + test_batch_size, len(test_pairs))
            
            # 准备批次数据
            batch_img1 = torch.stack([test_pairs[idx][0] for idx in range(i, end_idx)]).to(device)
            batch_img2 = torch.stack([test_pairs[idx][1] for idx in range(i, end_idx)]).to(device)
            batch_labels = torch.tensor([test_labels[idx] for idx in range(i, end_idx)], dtype=torch.float).unsqueeze(1).to(device)
            
            # 前向传播
            outputs = model(batch_img1, batch_img2)
            scores = torch.sigmoid(outputs)
            predicted = (scores > 0.5).float()
            
            # 记录预测结果
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(batch_labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # 计算评估指标
    test_preds = np.array(test_preds).flatten()
    test_targets = np.array(test_targets).flatten()
    all_scores = np.array(all_scores).flatten()
    
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds, average='binary')
    test_recall = recall_score(test_targets, test_preds, average='binary')
    test_f1 = f1_score(test_targets, test_preds, average='binary')
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # 8. 可视化结果
    print("\nGenerating visualizations...")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('results/siamese_network_training_curves.png')
    plt.show()
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_targets, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/siamese_network_confusion_matrix.png')
    plt.show()
    
    # 绘制分数分布
    plt.figure(figsize=(10, 6))
    same_scores = all_scores[test_targets == 1]
    diff_scores = all_scores[test_targets == 0]
    
    plt.hist(same_scores, bins=20, alpha=0.5, label='Same Person')
    plt.hist(diff_scores, bins=20, alpha=0.5, label='Different Person')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Similarity Scores')
    plt.savefig('results/siamese_network_score_distribution.png')
    plt.show()
    
    return model, (train_losses, val_losses, train_accs, val_accs)

if __name__ == '__main__':
    # 设置随机种子确保可重现性
    set_seed(42)
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Train models for fingerprint analysis')
    parser.add_argument('--data_dir', type=str, default='dataset/SOCOFing_Real/', help='Path to the dataset')
    parser.add_argument('--task', type=str, choices=['finger', 'gender', 'siamese', 'all'], default='all', 
                        help='Which task to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--use_resnet', action='store_true', default=True, help='Use ResNet model for finger classification')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    if args.task == 'finger' or args.task == 'all':
        print("\n" + "="*50)
        print("Training Finger Classifier (Task 3)")
        print("="*50)
        train_finger_classifier(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
            use_resnet=args.use_resnet,
            patience=args.patience
        )
    
    if args.task == 'gender' or args.task == 'all':
        print("\n" + "="*50)
        print("Training Gender Classifier (Task 4)")
        print("="*50)
        train_gender_classifier(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
            patience=args.patience
        )
    
    if args.task == 'siamese' or args.task == 'all':
        print("\n" + "="*50)
        print("Training Siamese Network for Fingerprint Verification (Task 2)")
        print("="*50)
        train_siamese_network(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
            patience=args.patience
        ) 