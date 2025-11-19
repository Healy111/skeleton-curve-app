import numpy as np
import matplotlib.pyplot as plt
import os
from 曲线比较可视化 import*

def visualize_hysteresis(hysteresis_data, title="Hysteresis Curve"):
    """
    可视化滞回曲线
    """
    plt.figure(figsize=(12, 8))

    # 绘制滞回曲线
    plt.scatter(hysteresis_data[:, 0], hysteresis_data[:, 1],
                s=1, alpha=0.7, color='blue', label='Hysteresis Curve')

    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加数据点统计信息
    plt.text(0.02, 0.98, f'Data points: {len(hysteresis_data)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

hysteresis_file = r"试验数据/arak20.txt"
print("正在读取曲线数据...")
hysteresis_data = read_hysteresis_data(hysteresis_file)
print("正在生成可视化图像...")
visualize_hysteresis(hysteresis_data,
                         "Hysteresis Curve - Test")

print("可视化完成！")

