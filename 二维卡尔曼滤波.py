import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os

class KalmanFilter2D:
    """
    二维卡尔曼滤波器实现
    状态向量包含 [x, y, vx, vy]，即位置和速度
    """
    def __init__(self, initial_state: np.ndarray, process_noise: float = 1e-1, measurement_noise: float = 1e-4):
        """
        初始化二维卡尔曼滤波器
        
        Args:
            initial_state: 初始状态 [x, y, vx, vy]
            process_noise: 过程噪声
            measurement_noise: 测量噪声
        """
        # 状态向量 [x, y, vx, vy]
        self.state = initial_state.astype(float)
        
        # 状态协方差矩阵
        self.P = np.eye(4) * 1.0
        
        # 状态转移矩阵（假设恒定速度模型）
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx*dt (假设dt=1)
            [0, 1, 0, 1],  # y = y + vy*dt
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0],  # 观测x位置
            [0, 1, 0, 0]   # 观测y位置
        ])
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(4) * process_noise
        
        # 测量噪声协方差矩阵
        self.R = np.eye(2) * measurement_noise
        
    def predict(self):
        """
        预测步骤
        """
        # 预测状态
        self.state = self.F @ self.state
        
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """
        更新步骤
        
        Args:
            measurement: 测量值 [x, y]
        """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态估计
        y = measurement - self.H @ self.state  # 残差
        self.state = self.state + K @ y
        
        # 更新协方差估计
        I = np.eye(len(self.state))
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[:2]  # 返回位置估计[x, y]

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

def apply_kalman_filter_2d(displacement: List[float], force: List[float], 
                          process_noise: float = 1e-4, measurement_noise: float = 1e-1) -> Tuple[List[float], List[float]]:
    """
    对二维数据应用卡尔曼滤波
    
    Args:
        displacement: 位移数据
        force: 力数据
        process_noise: 过程噪声
        measurement_noise: 测量噪声
        
    Returns:
        滤波后的位移和力数据
    """
    if len(displacement) == 0 or len(force) == 0:
        return [], []
    
    # 初始化滤波器
    initial_state = np.array([displacement[0], force[0], 0, 0])
    kf = KalmanFilter2D(initial_state, process_noise, measurement_noise)
    
    # 存储滤波结果
    filtered_displacement = [displacement[0]]
    filtered_force = [force[0]]
    
    # 对每个数据点进行滤波
    for i in range(1, len(displacement)):
        # 预测
        kf.predict()
        
        # 更新
        measurement = np.array([displacement[i], force[i]])
        estimated_state = kf.update(measurement)
        
        # 保存结果
        filtered_displacement.append(estimated_state[0])
        filtered_force.append(estimated_state[1])
        
    return filtered_displacement, filtered_force

# 读取骨架曲线数据
data_path = '骨架曲线原始数据/umecus原始骨架曲线.txt'
displacement, force = read_skeleton_data(data_path)

# 对数据应用二维卡尔曼滤波
filtered_displacement, filtered_force = apply_kalman_filter_2d(
    displacement, force, process_noise=1e-4, measurement_noise=1e-1
)

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

# 绘制二维卡尔曼滤波后的骨架曲线
plt.subplot(2, 2, 2)
plt.plot(filtered_displacement, filtered_force, linewidth=1.5, color='red', label='2D Kalman Filtered Curve')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('2D Kalman Filtered Skeleton Curve')
plt.grid(True, alpha=0.3)
plt.legend()

# 对比图
plt.subplot(2, 2, 3)
plt.plot(displacement, force, linewidth=1.0, color='blue', alpha=0.7, label='Original')
plt.plot(filtered_displacement, filtered_force, linewidth=1.5, color='red', label='2D Kalman Filtered')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Original vs 2D Kalman Filtered')
plt.grid(True, alpha=0.3)
plt.legend()

# 误差图
plt.subplot(2, 2, 4)
error = [np.sqrt((dx - dfx)**2 + (dy - dfy)**2) 
         for dx, dfx, dy, dfy in zip(displacement, filtered_displacement, force, filtered_force)]
plt.plot(range(len(error)), error, linewidth=1.0, color='green')
plt.xlabel('Data Point Index')
plt.ylabel('Euclidean Error')
plt.title('Euclidean Error Between Original and Filtered Data')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印一些统计信息
print(f"原始数据点数: {len(force)}")
print(f"滤波后数据点数: {len(filtered_force)}")
mse = np.mean([(orig - filt)**2 for orig, filt in zip(force, filtered_force)])
print(f"力数据均方误差: {mse:.6f}")
rmse = np.sqrt(np.mean([((dx - dfx)**2 + (dy - dfy)**2) 
                        for dx, dfx, dy, dfy in zip(displacement, filtered_displacement, force, filtered_force)]))
print(f"二维欧几里得均方根误差: {rmse:.6f}")
