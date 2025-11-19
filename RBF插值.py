import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import Rbf
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取外包络数据
file_path = 'F:/课题/计算机视觉识别屈服点项目/混合包络线法提取骨架曲线/外包络骨架曲线数据/806040原始骨架曲线_外包络数据.txt'
data = pd.read_csv(file_path, delimiter='\t')

# 不区分正负向，合并所有数据并按位移排序
data_sorted = data.sort_values('位移')
displacement = data_sorted['位移'].values
force = data_sorted['力'].values

# 使用径向基函数插值进行平滑处理
def rbf_smooth(x, y, function='multiquadric', smooth_factor=0.1, num_points=300):
    """
    使用径向基函数插值进行平滑处理
    参数:
    - x: x坐标数据
    - y: y坐标数据
    - function: RBF函数类型 ('multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate')
    - smooth_factor: 平滑因子，控制拟合程度
    - num_points: 插值点数量
    """
    # 创建插值点
    x_new = np.linspace(x.min(), x.max(), num_points)
    
    # 使用RBF进行插值
    rbf = Rbf(x, y, function=function, smooth=smooth_factor)
    y_smooth = rbf(x_new)
    
    return x_new, y_smooth
def save_smoothed_data(input_path, x_smooth, y_smooth):
    """保存平滑处理后的包络线数据"""
    import os

    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0].replace('_外包络数据', '')

    # 构建输出文件路径
    output_folder = 'F:/课题/计算机视觉识别屈服点项目/混合包络线法提取骨架曲线/外包络骨架曲线数据'
    os.makedirs(output_folder, exist_ok=True)

    # 保存平滑后的数据到一个文件
    smoothed_output_path = os.path.join(output_folder, f"{input_filename}_RBF平滑外包络数据.txt")

    with open(smoothed_output_path, 'w', encoding='utf-8') as f:
        f.write("位移\t力\n")

        # 写入平滑数据
        for i in range(len(x_smooth)):
            f.write(f"{x_smooth[i]:.6f}\t{y_smooth[i]:.6f}\n")

    print(f"RBF平滑外包络数据已保存至: {smoothed_output_path}")

def main_process():
    x_smooth, y_smooth = rbf_smooth(displacement, force, function='multiquadric', smooth_factor=0.05, num_points=300)
    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.scatter(displacement, force, c='blue', s=30, label='原始包络点', zorder=3)
    # 绘制径向基函数插值平滑曲线
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2.5, label='RBF平滑包络线')

    # 设置图表属性
    plt.xlabel('位移 (Displacement)')
    plt.ylabel('力 (Force)')
    plt.title('骨架曲线外包络线径向基函数插值平滑处理')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印平滑处理后的数据信息
    print("原始包络线数据点数:", len(displacement))
    print("平滑包络线数据点数:", len(x_smooth))
    # 保存平滑后的数据
    # save_smoothed_data(file_path, x_smooth, y_smooth)
if __name__ == "__main__":
    main_process()
