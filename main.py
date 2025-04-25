import os
import argparse
import torch
import numpy as np

from data_loader import get_data_loaders
from models import FingerCNN, GenderCNN, SiameseNetwork, ResNet18Fingerprint
from train import train_finger_classifier, train_gender_classifier, train_siamese_network, set_seed
from fingerprint_generator import train_fingerprint_generator

def main():
    parser = argparse.ArgumentParser(description='指纹分析与识别研究项目')
    parser.add_argument('--data_dir', type=str, default='dataset/SOCOFing_Real/', help='数据集路径')
    parser.add_argument('--task', type=str, choices=['generator', 'finger', 'gender', 'siamese', 'all'], 
                        default='all', help='要执行的任务')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--img_size', type=int, default=96, help='图像尺寸')
    parser.add_argument('--use_resnet', action='store_true', help='使用ResNet模型进行手指分类')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--target_finger', type=int, default=None, help='指纹生成任务中的目标手指索引(0-9)')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 设置随机种子以确保可重现性
    set_seed(args.seed)
    
    # 打印使用的设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

    print("\n数据集路径:", args.data_dir)
    
    # 执行指定的任务
    if args.task == 'generator' or args.task == 'all':
        print("\n" + "="*50)
        print("任务1: 训练指纹生成模型 - 通过9个手指生成第10个手指的指纹")
        print("="*50)
        train_fingerprint_generator(
            data_dir=args.data_dir,
            batch_size=8 if args.batch_size > 8 else args.batch_size,  # 生成任务的batch_size通常需要较小
            epochs=args.epochs,
            learning_rate=0.0005 if args.lr == 0.001 else args.lr,  # 生成任务通常需要更小的学习率
            img_size=args.img_size,
            model_save_path='models/fingerprint_generator.pth',
            target_finger=args.target_finger
        )
    
    if args.task == 'finger' or args.task == 'all':
        print("\n" + "="*50)
        print("任务3: 训练手指识别模型 - 确定指纹来自哪根手指")
        print("="*50)
        train_finger_classifier(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
            model_save_path='models/finger_classifier.pth',
            use_resnet=args.use_resnet
        )
    
    if args.task == 'gender' or args.task == 'all':
        print("\n" + "="*50)
        print("任务4: 训练性别识别模型 - 确定指纹来自什么性别")
        print("="*50)
        train_gender_classifier(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
            model_save_path='models/gender_classifier.pth'
        )
    
    if args.task == 'siamese' or args.task == 'all':
        print("\n" + "="*50)
        print("任务2: 训练孪生网络 - 鉴别两个指纹是否来自同一个人")
        print("="*50)
        train_siamese_network(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
            model_save_path='models/siamese_network.pth'
        )

if __name__ == '__main__':
    main() 