# 指纹机器学习研究项目

这个项目使用机器学习技术对指纹图像进行分析、识别和生成。项目包含四个主要任务：

1. **指纹生成**：通过9个手指的指纹图像，生成第10个手指的指纹图像
2. **指纹验证**：判断两个指纹是否来自同一个人
3. **手指分类**：确定一个指纹来自哪一根手指
4. **性别分类**：确定一个指纹来自什么性别

## 环境要求

- Python 3.10
- PyTorch (with CUDA)
- 其他依赖项见 `requirements.txt`

## 安装

```bash
# 克隆仓库
git clone https://github.com/Gloridust/fingerprint_ML_research.git
cd fingerprint_ML_research

# 安装依赖
pip install -r requirements.txt
```

## 数据集

本项目使用SOCOFing数据集，请将数据放在 `dataset/SOCOFing_Real/` 目录下。

文件命名格式示例: `101__M_Right_middle_finger.BMP`，其中：
- `101` 是个体ID
- `M` 表示性别 (M: 男性, F: 女性)
- `Right` 表示手 (Left/Right)
- `middle` 表示手指名称 (thumb/index/middle/ring/little)

## 使用方法

### 运行所有任务
```bash
python main.py --data_dir dataset/SOCOFing_Real/ --task all --epochs 30
```

### 单独运行指纹生成任务（任务1）
```bash
python main.py --task generator --epochs 50 --lr 0.0005

# 指定生成右手拇指的指纹 (索引5)
python main.py --task generator --target_finger 5
```

### 单独运行指纹验证任务（任务2）
```bash
python main.py --task siamese --epochs 40
```

### 单独运行手指分类任务（任务3）
```bash
python main.py --task finger 

# 使用预训练的ResNet模型进行特征提取
python main.py --task finger --use_resnet
```

### 单独运行性别分类任务（任务4）
```bash
python main.py --task gender
```

## 命令行参数

- `--data_dir`: 数据集路径 (默认: 'dataset/SOCOFing_Real/')
- `--task`: 要执行的任务 ['generator', 'finger', 'gender', 'siamese', 'all'] (默认: 'all')
- `--batch_size`: 批次大小 (默认: 32)
- `--epochs`: 训练轮数 (默认: 30)
- `--lr`: 学习率 (默认: 0.001)
- `--img_size`: 图像大小 (默认: 96)
- `--use_resnet`: 是否使用ResNet模型进行手指分类 (默认: False)
- `--seed`: 随机种子 (默认: 42)
- `--target_finger`: 指纹生成任务中的目标手指索引 (默认: None，随机选择)

## 项目结构

- `data_loader.py`: 数据加载和预处理模块
- `models.py`: 各种模型架构定义
- `train.py`: 训练、验证和测试逻辑
- `fingerprint_generator.py`: 指纹生成模型实现
- `main.py`: 主程序入口

## 结果

训练过程和结果将输出到控制台，并保存到以下目录：

- 模型权重: `models/`
- 可视化结果: `results/`

## 注意事项

- 指纹生成任务对计算资源要求较高，建议使用GPU进行训练
- 为了获得最佳结果，建议使用完整的SOCOFing数据集，并确保每个个体有10个手指的数据 