import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from scipy import signal
import os


def read_skeleton_data(file_path: str) -> Tuple[List[float], List[float]]:
    """
    读取骨架曲线数据

    Args:
        file_path: 数据文件路径

    Returns:
        位移和力数据列表
    """
    displacements = []
    forces = []

    # 使用 utf-8 编码读取文件，添加错误处理
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    # 如果 utf-8 失败，尝试使用 gbk 编码
    if not lines:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
            lines = file.readlines()

    # 跳过标题行
    for line in lines[1:]:
        line = line.strip()
        if line:
            try:
                values = line.split('\t')
                if len(values) >= 2:
                    displacement = float(values[0])
                    force = float(values[1])
                    displacements.append(displacement)
                    forces.append(force)
            except ValueError:
                continue

    return displacements, forces


def apply_lowpass_filter(data: List[float], cutoff_freq: float = 0.1, sampling_rate: float = 1.0,
                        filter_order: int = 4) -> List[float]:
    """
    应用低通滤波器到数据
    
    Args:
        data: 输入数据
        cutoff_freq: 截止频率 (相对于奈奎斯特频率的比例)
        sampling_rate: 采样率
        filter_order: 滤波器阶数
        
    Returns:
        滤波后的数据
    """
    # 设计巴特沃斯低通滤波器
    nyquist = sampling_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist
    
    # 确保截止频率在有效范围内
    normalized_cutoff = min(normalized_cutoff, 0.99)
    normalized_cutoff = max(normalized_cutoff, 0.01)
    
    # 创建滤波器系数
    b, a = signal.butter(filter_order, normalized_cutoff, btype='low', analog=False)
    
    # 应用滤波器
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data.tolist()

# 读取骨架曲线数据
data_path = '外包络骨架曲线数据/806040原始骨架曲线_外包络数据.txt'
displacement, force = read_skeleton_data(data_path)

# 对力数据应用低通滤波
# 可以调整 cutoff_freq 参数来控制滤波强度
filtered_force = apply_lowpass_filter(force, cutoff_freq=0.2, sampling_rate=1.0, filter_order=2)

# 可视化处理结果
plt.figure(figsize=(15, 10))

# 绘制原始骨架曲线
plt.subplot(2, 2, 1)
plt.plot(displacement, force, linewidth=1.5, color='blue', alpha=0.7, label='Original Skeleton Curve')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Original Skeleton Curve')
plt.grid(True, alpha=0.3)
plt.legend()

# 绘制低通滤波后的骨架曲线
plt.subplot(2, 2, 2)
plt.plot(displacement, filtered_force, linewidth=1.5, color='red', label='Lowpass Filtered Curve')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Lowpass Filtered Skeleton Curve')
plt.grid(True, alpha=0.3)
plt.legend()

# 对比图
plt.subplot(2, 2, 3)
plt.plot(displacement, force, linewidth=1.0, color='blue', alpha=0.7, label='Original')
plt.plot(displacement, filtered_force, linewidth=1.5, color='red', label='Lowpass Filtered')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Original vs Lowpass Filtered')
plt.grid(True, alpha=0.3)
plt.legend()

# 误差图
plt.subplot(2, 2, 4)
error = [abs(orig - filt) for orig, filt in zip(force, filtered_force)]
plt.plot(displacement, error, linewidth=1.0, color='green')
plt.xlabel('Displacement')
plt.ylabel('Absolute Error')
plt.title('Absolute Error Between Original and Filtered Data')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印一些统计信息
print(f"原始数据点数: {len(force)}")
print(f"滤波后数据点数: {len(filtered_force)}")
mse = np.mean([(orig - filt)**2 for orig, filt in zip(force, filtered_force)])
print(f"均方误差: {mse:.6f}")

