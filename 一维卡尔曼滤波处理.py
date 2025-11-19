import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os

class KalmanFilter:
    """
    一维卡尔曼滤波器实现
    """
    def __init__(self, process_variance: float, measurement_variance: float, initial_value: float = 0):
        self.process_variance = process_variance  # 过程噪声方差
        self.measurement_variance = measurement_variance  # 测量噪声方差
        self.posteri_estimate = initial_value  # 后验估计
        self.posteri_error_estimate = 1.0  # 后验误差估计
        
    def update(self, measurement: float) -> float:
        """
        卡尔曼滤波更新步骤
        
        Args:
            measurement: 测量值
            
        Returns:
            滤波后的估计值
        """
        # 预测步骤
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # 更新步骤
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        
        return self.posteri_estimate

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
    
    with open(file_path, 'r') as file:
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

def apply_kalman_filter(data: List[float], process_variance: float = 1e-3, 
                       measurement_variance: float = 1e-1) -> List[float]:
    """
    对数据应用卡尔曼滤波
    
    Args:
        data: 输入数据
        process_variance: 过程噪声方差
        measurement_variance: 测量噪声方差
        
    Returns:
        滤波后的数据
    """
    kf = KalmanFilter(process_variance, measurement_variance, data[0])
    filtered_data = [kf.update(point) for point in data]
    return filtered_data

# 读取骨架曲线数据
data_path = '骨架曲线原始数据/abouorc2原始骨架曲线.txt'
displacement, force = read_skeleton_data(data_path)

# 对力数据应用卡尔曼滤波
filtered_force = apply_kalman_filter(force, process_variance=1e-3, measurement_variance=1e-1)

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

# 绘制卡尔曼滤波后的骨架曲线
plt.subplot(2, 2, 2)
plt.plot(displacement, filtered_force, linewidth=1.5, color='red', label='Kalman Filtered Curve')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Kalman Filtered Skeleton Curve')
plt.grid(True, alpha=0.3)
plt.legend()

# 对比图
plt.subplot(2, 2, 3)
plt.plot(displacement, force, linewidth=1.0, color='blue', alpha=0.7, label='Original')
plt.plot(displacement, filtered_force, linewidth=1.5, color='red', label='Kalman Filtered')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Original vs Kalman Filtered')
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
