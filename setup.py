import os
import argparse
import sys

def setup_directories():
    """
    创建项目所需的目录结构
    """
    # 创建必要的目录
    dirs = [
        'models',
        'results',
        'dataset/SOCOFing_Real'
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            print(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            print(f"目录已存在: {dir_path}")
    
    # 检查数据集目录是否有文件
    dataset_dir = 'dataset/SOCOFing_Real'
    if not os.path.exists(dataset_dir):
        print(f"警告: 数据集目录 {dataset_dir} 不存在，已创建空目录")
    else:
        files = os.listdir(dataset_dir)
        if not files:
            print(f"警告: 数据集目录 {dataset_dir} 为空，请添加指纹图像")
        else:
            print(f"数据集目录 {dataset_dir} 包含 {len(files)} 个文件")
            
            # 检查文件命名格式
            sample_file = files[0]
            if not (sample_file.endswith('.BMP') and '__' in sample_file):
                print(f"警告: 文件命名格式可能不正确。示例文件: {sample_file}")
                print("期望格式: <SubjectID>__<Gender>_<Hand>_<FingerName>_finger.BMP")
            else:
                print(f"文件命名格式正确。示例文件: {sample_file}")

def check_environment():
    """
    检查Python环境和必要的库
    """
    print(f"Python版本: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("警告: 未找到PyTorch。请安装: pip install torch torchvision")
    
    try:
        import numpy
        print(f"NumPy版本: {numpy.__version__}")
    except ImportError:
        print("警告: 未找到NumPy。请安装: pip install numpy")
    
    try:
        import cv2
        print(f"OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("警告: 未找到OpenCV。请安装: pip install opencv-python")
    
    try:
        import matplotlib
        print(f"Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("警告: 未找到Matplotlib。请安装: pip install matplotlib")
    
    try:
        import sklearn
        print(f"Scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("警告: 未找到Scikit-learn。请安装: pip install scikit-learn")
    
    try:
        import pandas
        print(f"Pandas版本: {pandas.__version__}")
    except ImportError:
        print("警告: 未找到Pandas。请安装: pip install pandas")
    
    try:
        from skimage.metrics import structural_similarity
        print("Scikit-image (用于SSIM): 已安装")
    except ImportError:
        print("警告: 未找到Scikit-image。请安装: pip install scikit-image")

def main():
    parser = argparse.ArgumentParser(description='指纹机器学习研究项目 - 环境设置')
    parser.add_argument('--check_env', action='store_true', help='检查Python环境和必要的库')
    
    args = parser.parse_args()
    
    # 创建目录结构
    setup_directories()
    
    # 检查环境
    if args.check_env:
        print("\n检查Python环境和必要的库...")
        check_environment()
    
    print("\n设置完成！可以通过以下命令运行项目:")
    print("python main.py --data_dir dataset/SOCOFing_Real/ --task all")

if __name__ == '__main__':
    main() 