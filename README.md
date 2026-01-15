# SAM-LoRA: 基于LoRA的SAM医学图像分割微调框架

## 📋 项目简介

本项目实现了基于 **LoRA (Low-Rank Adaptation)** 方法对 **SAM (Segment Anything Model)** 进行高效微调，专门用于医学图像分割任务。通过在SAM的图像编码器注意力层中注入低秩矩阵，在保持原始模型大部分参数冻结的情况下，实现对特定医学图像数据集的高效适应。

### 主要特性

- 🚀 **高效微调**: 仅训练LoRA注入的参数，大幅减少计算资源需求
- 🏥 **医学图像专用**: 针对CT、MRI等医学影像进行优化
- 📊 **多种评估指标**: 支持Dice系数和Surface Dice评估
- 🎯 **灵活提示方式**: 支持框提示(Box)、点提示(Point)及混合提示
- 💾 **权重管理**: 支持safetensors格式保存和加载LoRA权重

## 📁 项目结构

```
LoRA/
├── train_LoRA.py          # 训练脚本
├── test_LoRA.py           # 测试脚本
├── README.md              # 项目说明文档
├── src/                   # 核心源代码
│   ├── __init__.py
│   ├── lora.py            # LoRA模块实现
│   ├── dataloader.py      # 数据加载器
│   ├── processor.py       # 输入处理器
│   ├── utils.py           # 工具函数
│   └── segment_anything/  # SAM模型代码
│       ├── __init__.py
│       ├── build_sam.py   # SAM模型构建
│       ├── predictor.py   # SAM预测器
│       ├── automatic_mask_generator.py
│       ├── modeling/      # 模型组件
│       │   ├── sam.py
│       │   ├── image_encoder.py
│       │   ├── mask_decoder.py
│       │   ├── prompt_encoder.py
│       │   └── transformer.py
│       └── utils/         # SAM工具
│           ├── transforms.py
│           └── amg.py
└── utils/                 # 数据处理工具
    ├── README.md          # 数据处理说明
    ├── split.py           # 数据集划分
    ├── pre_CT_MR.py       # CT/MR预处理
    ├── pre_grey_rgb.py    # 灰度/RGB预处理
    ├── format_convert.py  # 格式转换
    ├── ckpt_convert.py    # 检查点转换
    ├── SurfaceDice.py     # Surface Dice指标
    └── demo.py            # 演示脚本
```

## 🔧 环境配置

### 依赖安装

```bash
pip install torch torchvision
pip install monai
pip install safetensors
pip install numpy
pip install matplotlib
pip install Pillow
pip install scikit-learn
pip install tqdm
pip install pyyaml
```

### 硬件要求

- GPU: 建议NVIDIA GPU，显存 >= 8GB
- 支持CUDA加速

## 📊 数据准备

### 数据格式

数据应预处理为 `.npz` 格式，包含以下键值：
- `imgs`: 图像数组，形状为 `(N, H, W, 3)` (RGB图像)
- `gts`: 标签数组，形状为 `(N, H, W)` (二值掩码)

### 数据组织结构

```
dataset/
├── train/
│   ├── case1.npz
│   ├── case2.npz
│   └── ...
└── test/
    ├── case1.npz
    └── ...
```

### 数据预处理

使用 `utils/` 目录下的工具进行数据预处理：

```bash
# 数据格式转换
python utils/format_convert.py

# CT/MR数据预处理
python utils/pre_CT_MR.py

# 灰度/RGB图像预处理
python utils/pre_grey_rgb.py

# 数据集划分
python utils/split.py
```

## 🏋️ 模型训练

### 配置训练参数

编辑 `train_LoRA.py` 中的关键参数：

```python
# 数据路径
train_dataset_path = "dataset/train"

# SAM预训练权重路径
checkpoint = "path/to/sam_vit_b_01ec64.pth"

# 工作目录
work_dir = 'work_dir/your_experiment'
task_name = 'experiment_name'

# LoRA秩
rank = 4

# 训练超参数
num_epochs = 10
batch_size = 1
learning_rate = 1e-3
accumulation_steps = 8  # 梯度累积步数
```

### 启动训练

```bash
python train_LoRA.py
```

### 训练输出

训练过程会在 `work_dir/{task_name}-{timestamp}/` 下保存：
- `medsam_model_best.pth`: 最佳模型权重
- `medsam_model_latest.pth`: 最新模型权重
- `train_loss.png`: 训练损失曲线图
- 训练脚本备份

## 🧪 模型测试

### 配置测试参数

编辑 `test_LoRA.py` 中的参数：

```python
# 测试数据路径
npz_ts_path = 'path/to/test_data'

# SAM原始权重路径
checkpoint = "path/to/sam_vit_b_01ec64.pth"

# 训练好的模型路径
model_path = "path/to/medsam_model_best.pth"

# 输出目录
output_dir = 'path/to/output'

# 设备
device = "cuda:0"
```

### 运行测试

```bash
python test_LoRA.py
```

### 评估指标

- **Dice Coefficient**: 衡量预测与真实标签的重叠程度
- **Surface Dice**: 评估边界分割质量

## 🔬 核心模块说明

### LoRA模块 (`src/lora.py`)

#### LoRA_qkv 类
在注意力机制的QKV层中注入LoRA适配器：

```python
class LoRA_qkv(nn.Module):
    """
    LoRA适配器用于注意力模块
    仅对queries(Q)和values(V)进行低秩适应
    
    参数:
        qkv: 原始注意力块
        linear_a_q, linear_b_q: Q的低秩矩阵
        linear_a_v, linear_b_v: V的低秩矩阵
    """
```

#### LoRA_sam 类
将LoRA注入SAM的图像编码器：

```python
class LoRA_sam(nn.Module):
    """
    将LoRA权重添加到SAM图像编码器的注意力块中
    
    参数:
        sam_model: SAM模型实例
        rank: LoRA矩阵的秩
        lora_layer: 需要应用LoRA的层列表
    """
```

主要方法：
- `save_lora_parameters(filename)`: 保存LoRA权重为safetensors格式
- `load_lora_parameters(filename)`: 加载LoRA权重

### 数据加载器 (`src/dataloader.py`)

```python
class DatasetSegmentation(Dataset):
    """
    医学图像分割数据集
    
    功能:
        - 加载npz格式数据
        - 自动生成边界框和点提示
        - 图像预处理
    """
```

### 处理器 (`src/processor.py`)

```python
class Samprocessor:
    """
    SAM输入处理器
    
    支持的提示模式:
        - "no": 无提示
        - "box": 仅框提示
        - "full_box": 全框提示
        - "pt_box": 点+框提示(默认)
    """
```

## 📈 支持的SAM变体

| 模型 | 编码器维度 | 编码器深度 | 注意力头数 |
|------|-----------|-----------|-----------|
| ViT-B | 768 | 12 | 12 |
| ViT-L | 1024 | 24 | 16 |
| ViT-H | 1280 | 32 | 16 |

## 🎛️ 参数说明

### LoRA超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| rank | LoRA矩阵的秩 | 4 |
| lora_layer | 应用LoRA的层 | 所有注意力层 |

### 训练超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| batch_size | 批次大小 | 1 |
| learning_rate | 学习率 | 1e-3 |
| num_epochs | 训练轮数 | 10 |
| accumulation_steps | 梯度累积步数 | 8 |
| weight_decay | 权重衰减 | 0 |

### 损失函数

使用 **DiceCELoss** (来自MONAI)：
- 结合Dice Loss和Cross Entropy Loss
- `sigmoid=True`: 输出经过sigmoid激活
- `squared_pred=True`: 使用平方预测值
- `reduction='mean'`: 损失取均值

## 📝 注意事项

1. **显存管理**: 
   - 使用梯度累积减少显存占用
   - 代码中包含 `torch.cuda.empty_cache()` 主动释放显存
   
2. **混合精度训练**:
   - 代码中包含注释的混合精度训练代码
   - 可根据需要启用以进一步减少显存使用

3. **可训练参数**:
   - 仅包含 "linear" 的参数会被解冻训练
   - 训练前会打印可训练参数数量

## 📚 参考资料

- [SAMed: Customized Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.13785) - 本项目的核心参考论文，提出了使用LoRA微调SAM进行医学图像分割的方法
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [MONAI: Medical Open Network for AI](https://monai.io/)

