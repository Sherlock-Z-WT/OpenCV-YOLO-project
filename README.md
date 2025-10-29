# 智能高级车辆检测与计数系统
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg">
  <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green.svg">
  <img src="https://img.shields.io/badge/YOLOv8-v8-orange.svg">
  <img src="https://img.shields.io/badge/PyTorch-1.10%2B-red.svg">
</div>

### 项目概述

这是一个基于深度学习的高级车辆检测与计数系统，利用YOLO (You Only Look Once) 目标检测算法和混合跟踪策略，实现对视频中车辆的精确检测、跟踪和计数。系统能够区分车辆进入和离开的方向，适用于交通流量分析、车流量统计和智能交通监控等场景。

### 核心特性

- **高精度车辆检测**：基于YOLOv5/YOLOv8模型，支持多种车辆类型（汽车、摩托车、公交车、卡车）
- **智能跟踪算法**：实现基于IoU和距离的混合匹配策略，提高跟踪稳定性
- **方向感知计数**：通过检测线技术，精确区分车辆进入(ENTER)和离开(EXIT)方向
- **多级过滤机制**：包含置信度过滤、尺寸过滤、区域过滤等，有效减少误报
- **内存优化**：实现自动清理机制，避免长时间运行时的内存溢出
- **实时可视化**：直观显示检测结果、车辆类型、计数信息和检测线

## 系统架构

系统采用模块化设计，由四个核心组件组成，实现了检测、跟踪、计数和视频处理的完整流程：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  VideoProcessor │────▶│  YOLODetector   │────▶│ MovementTracker │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲                                             │
         │                                             │
         └─────────────────────────────────────────────┘
```

### 组件说明

| 组件 | 职责 | 文件 |
|------|------|------|
| VideoProcessor | 视频读取、处理和输出管理 | <mcfile name="video_processor.py" path="video_processor.py"></mcfile> |
| YOLODetector | 车辆检测、类型识别和多级过滤 | <mcfile name="detector.py" path="detector.py"></mcfile> |
| MovementTracker | 车辆跟踪、方向判断和计数 | <mcfile name="tracker.py" path="tracker.py"></mcfile> |
| Main | 系统集成和工作流控制 | <mcfile name="main.py" path="main.py"></mcfile> |

## 技术实现细节

### 1. 车辆检测算法

系统使用Ultralytics YOLO框架，支持YOLOv5和YOLOv8模型，通过多级过滤机制提高检测准确性：

- **置信度过滤**：设置最小置信度阈值(0.35)过滤低置信度检测
- **类别过滤**：仅检测车辆类别(ID: 2, 3, 5, 7)
- **尺寸过滤**：基于最小宽度、高度和面积过滤过小的目标
- **形状过滤**：通过宽高比(0.3-2.5)排除异常形状
- **区域过滤**：定义ROI(感兴趣区域)，集中处理有效区域内的目标
- **位置过滤**：基于目标在画面中的位置和置信度进行综合判断

### 2. 智能跟踪机制

<mcfile name="tracker.py" path="tracker.py"></mcfile>中的<mcsymbol name="_update_tracked_objects" filename="tracker.py" path="tracker.py" startline="120" type="function"></mcsymbol>方法实现了高级混合匹配策略：

- **混合评分系统**：IoU(交并比)占60%，中心点距离占40%，提高匹配准确性
- **多帧历史跟踪**：维护5帧的位置历史，用于轨迹分析
- **匹配分阶段处理**：
  1. 优先匹配已有跟踪对象，确保轨迹连续性
  2. 为未匹配的检测创建新跟踪对象
  3. 清理不活跃或已计数的对象，优化内存使用

### 3. 方向感知计数系统

- **检测线设置**：可配置的检测线位置，默认位于视频底部10%处
- **过线检测算法**：基于历史位置变化，判断车辆是否穿过检测线
- **方向判断**：根据过线前后的位置变化，确定是进入(从上到下)还是离开(从下到上)
- **去重机制**：使用`counted_vehicles`集合确保每个车辆只被计数一次

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- Ultralytics YOLO
- NumPy

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/Sherlock-Z-WT/OpenCV-YOLO-project.git
cd OpenCV-YOLO-project
```

2. 创建并激活虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖包
```bash
pip install opencv-python torch torchvision numpy ultralytics
```

4. 下载预训练模型（可选，项目已包含轻量模型）
```bash
# 对于YOLOv8n (默认使用)
# 对于YOLOv5n
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

## 使用方法

### 基本使用

1. 修改<mcfile name="main.py" path="main.py"></mcfile>中的视频路径
```python
video_path = 'shenzhencar.mp4'  # 替换为你的视频文件路径
```

2. 运行主程序
```bash
python main.py
```

3. 处理结果将保存在`output/result.mp4`中

### 参数配置

主要配置参数位于各个组件的初始化方法中：

#### 检测器配置 (<mcfile name="detector.py" path="detector.py"></mcfile>)

```python
# 修改检测器参数
detector = YOLODetector(model_name='yolov8n.pt')
# 可自定义模型: 'yolov5n.pt', 'yolov5nu.pt', 'yolov8n.pt' 等
```

#### 跟踪器配置 (<mcfile name="tracker.py" path="tracker.py"></mcfile>)

```python
# 修改检测线位置 (默认0.9，距离底部10%)
tracker = MovementTracker(line_ratio=0.9)
```

## 性能优化

1. **模型选择**：根据需要选择不同精度/速度平衡的模型
   - `yolov5n.pt`/`yolov8n.pt`: 最轻量，速度最快
   - `yolov5s.pt`/`yolov8s.pt`: 中等性能
   - `yolov5m.pt`/`yolov8m.pt`: 更高精度，但速度较慢

2. **参数调优**：
   - 调整`confidence_threshold`平衡精度和召回率
   - 调整`max_track_distance`适应不同视频的车辆速度
   - 调整`line_ratio`优化计数线位置

3. **ROI设置**：在<mcfile name="detector.py" path="detector.py"></mcfile>中修改ROI参数，聚焦于需要监控的区域

## 项目结构

```
├── main.py                  # 主程序入口
├── detector.py              # YOLO检测器实现
├── tracker.py               # 车辆跟踪与计数
├── video_processor.py       # 视频处理工具
├── utils.py                 # 工具函数
├── config.yaml              # 配置文件
├── output/                  # 输出结果目录
│   └── result.mp4           # 处理后的视频
├── yolov5n.pt               # 预训练模型
├── yolov5nu.pt              # 预训练模型 (FP16量化)
├── yolov8n.pt               # 预训练模型
└── README.md                # 项目说明文档
```

## 应用场景

- **交通流量分析**：实时统计道路车辆数量和流动方向
- **停车场管理**：监控车辆进出，估算车位使用率
- **智能安防**：检测异常车辆流动模式
- **交通信号优化**：为交通灯配时提供数据支持
- **拥堵预警**：基于车流量变化预测可能的拥堵

## 进阶开发

### 扩展车辆类型

要增加支持的车辆类型，修改<mcfile name="tracker.py" path="tracker.py"></mcfile>中的`vehicle_types`映射和<mcfile name="detector.py" path="detector.py"></mcfile>中的`vehicle_classes`列表。

### 添加自定义过滤规则

在<mcfile name="detector.py" path="detector.py"></mcfile>的`detect`方法中添加新的过滤逻辑，或在<mcfile name="tracker.py" path="tracker.py"></mcfile>中增强跟踪算法。

### 多检测线支持

扩展<mcfile name="tracker.py" path="tracker.py"></mcfile>支持多个检测线，实现更复杂的交通流分析。

## 故障排除

### 常见问题

1. **检测不准确**
   - 调整`confidence_threshold`参数
   - 检查视频质量，确保光线充足
   - 尝试更高级别的模型

2. **跟踪丢失**
   - 增加`max_track_distance`值
   - 检查`position_history_length`参数

3. **内存占用过高**
   - 减小`cleanup_interval`值
   - 降低`max_inactive_time`参数

4. **计数不准确**
   - 调整检测线位置(`line_ratio`)
   - 检查`_check_line_crossing`方法的过线判断逻辑

## 许可证

[MIT License](LICENSE)

## 鸣谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 提供强大的目标检测模型
- [OpenCV](https://opencv.org/) - 开源计算机视觉库
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 贡献指南

欢迎提交Issue和Pull Request！在贡献代码前，请确保你的修改与项目整体风格一致，并通过基本测试。
