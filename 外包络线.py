import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.font_manager as fm
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题



def improved_geometric_filter(points, side='positive'):
    """改进的几何特性包络点筛选，考虑骨架曲线力绝对值先增后减的特点"""
    if len(points) < 3:
        return points

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    if side == 'positive':
        # 正向包络：按位移排序
        sorted_indices = np.argsort(hull_points[:, 0])
        sorted_points = hull_points[sorted_indices]

        # 找到力绝对值最大的点作为转折点
        abs_forces = np.abs(sorted_points[:, 1])
        peak_index = np.argmax(abs_forces)

        # 分两段处理：上升段和下降段
        result_points = []

        # 上升段：从左到转折点，力绝对值应递增
        if peak_index > 0:
            ascending_points = sorted_points[:peak_index + 1]
            result_points.append(ascending_points[0])

            for i in range(1, len(ascending_points)):
                current = ascending_points[i]
                prev = result_points[-1]

                # 检查力绝对值是否增长（允许小误差）
                if np.abs(current[1]) >= np.abs(prev[1]) - 0.01 * np.abs(prev[1]):
                    # 避免过于接近的点
                    if abs(current[0] - prev[0]) > 1e-6 or abs(current[1] - prev[1]) > 1e-6:
                        result_points.append(current)

        # 下降段：从转折点到右端，力绝对值应递减
        if peak_index < len(sorted_points) - 1:
            descending_points = sorted_points[peak_index:]

            # 如果上升段没有点，则添加转折点
            if len(result_points) == 0:
                result_points.append(descending_points[0])
            # 否则检查转折点是否已在结果中
            elif not np.allclose(result_points[-1], descending_points[0]):
                result_points.append(descending_points[0])

            for i in range(1, len(descending_points)):
                current = descending_points[i]
                prev = result_points[-1]

                # 检查力绝对值是否减少（允许小误差）
                if np.abs(current[1]) <= np.abs(prev[1]) + 0.01 * np.abs(prev[1]):
                    # 避免过于接近的点
                    if abs(current[0] - prev[0]) > 1e-6 or abs(current[1] - prev[1]) > 1e-6:
                        result_points.append(current)

        return np.array(result_points) if len(result_points) > 0 else np.array([])

    else:  # negative side
        # 负向包络：按位移排序（从左到右）
        sorted_indices = np.argsort(hull_points[:, 0])
        sorted_points = hull_points[sorted_indices]

        # 对于负向，通常x为负值，找到力绝对值最大的点作为转折点
        abs_forces = np.abs(sorted_points[:, 1])
        peak_index = np.argmax(abs_forces)

        # 分两段处理
        result_points = []

        # 上升段：从左到转折点，力绝对值应递增
        if peak_index > 0:
            ascending_points = sorted_points[:peak_index + 1]
            result_points.append(ascending_points[0])

            for i in range(1, len(ascending_points)):
                current = ascending_points[i]
                prev = result_points[-1]

                # 检查力绝对值是否增长
                if np.abs(current[1]) >= np.abs(prev[1]) - 0.01 * np.abs(prev[1]):
                    if abs(current[0] - prev[0]) > 1e-6 or abs(current[1] - prev[1]) > 1e-6:
                        result_points.append(current)

        # 下降段：从转折点到右端，力绝对值应递减
        if peak_index < len(sorted_points) - 1:
            descending_points = sorted_points[peak_index:]

            if len(result_points) == 0:
                result_points.append(descending_points[0])
            elif not np.allclose(result_points[-1], descending_points[0]):
                result_points.append(descending_points[0])

            for i in range(1, len(descending_points)):
                current = descending_points[i]
                prev = result_points[-1]

                # 检查力绝对值是否减少
                if np.abs(current[1]) <= np.abs(prev[1]) + 0.01 * np.abs(prev[1]):
                    if abs(current[0] - prev[0]) > 1e-6 or abs(current[1] - prev[1]) > 1e-6:
                        result_points.append(current)

        return np.array(result_points) if len(result_points) > 0 else np.array([])
# 保存关键点数据到txt文件
def save_envelope_data(input_path, positive_points, negative_points):
    """保存包络线关键点数据到一个txt文件，按位移顺序排列"""
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]

    # 构建输出文件路径
    output_folder = 'F:/课题/计算机视觉识别屈服点项目/混合包络线法提取骨架曲线/外包络骨架曲线数据'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 合并正负向包络线数据
    all_points = []

    # 添加正向包络线数据
    if len(positive_points) > 0:
        for point in positive_points:
            all_points.append([point[0], point[1]])

    # 添加负向包络线数据
    if len(negative_points) > 0:
        for point in negative_points:
            all_points.append([point[0], point[1]])

    # 转换为numpy数组并按位移排序
    if len(all_points) > 0:
        all_points = np.array(all_points)
        sorted_indices = np.argsort(all_points[:, 0])
        all_points = all_points[sorted_indices]

    # 保存合并后的数据到一个文件
    combined_output_path = os.path.join(output_folder, f"{input_filename}_外包络数据.txt")

    with open(combined_output_path, 'w', encoding='utf-8') as f:
        f.write("位移\t力\n")

        # 写入按位移排序的所有包络线数据
        if len(all_points) > 0:
            for point in all_points:
                f.write(f"{point[0]:.6f}\t{point[1]:.6f}\n")

    print(f"外包络数据已保存至: {combined_output_path}")
def main_process():
    # 输入文件路径
    input_file_path = 'F:/课题/计算机视觉识别屈服点项目/混合包络线法提取骨架曲线/骨架曲线原始数据/806040原始骨架曲线.txt'

    # 读取数据
    data = pd.read_csv(input_file_path, delimiter='\t', skiprows=1, names=['Displacement', 'Force'])

    # 提取位移和力数据
    displacement = data['Displacement'].values
    force = data['Force'].values

    # 绘制原始数据
    plt.figure(figsize=(12, 8))
    plt.plot(displacement, force, 'b-', linewidth=1, alpha=0.7, label='原始骨架曲线')

    # 分离正负位移数据
    positive_indices = displacement >= 0
    negative_indices = displacement <= 0  # 包含0点

    positive_points = np.column_stack((displacement[positive_indices], force[positive_indices]))
    negative_points = np.column_stack((displacement[negative_indices], force[negative_indices]))
    # 重新计算包络线
    if len(positive_points) > 2:
        positive_envelope = improved_geometric_filter(positive_points, 'positive')
        if len(positive_envelope) > 0:
            plt.plot(positive_envelope[:, 0], positive_envelope[:, 1], 'r-', linewidth=2.5,
                     label='正向凸包络线')

    if len(negative_points) > 2:
        negative_envelope = improved_geometric_filter(negative_points, 'negative')
        if len(negative_envelope) > 0:
            plt.plot(negative_envelope[:, 0], negative_envelope[:, 1], 'g-', linewidth=2.5,
                     label='负向凸包络线')

    # 标记关键点
    plt.scatter(displacement[0], force[0], color='blue', s=50, zorder=5, label='起始点')
    plt.scatter(displacement[-1], force[-1], color='purple', s=50, zorder=5, label='终点')

    # 添加图例和标签
    plt.xlabel('位移 (Displacement)')
    plt.ylabel('力 (Force)')
    plt.title('骨架曲线及其正负向凸包络线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 显示图形
    plt.tight_layout()
    plt.show()

    # 打印包络线的关键点信息
    print("正向凸包络线关键点:")
    if 'positive_envelope' in locals():
        for i in range(len(positive_envelope)):
            print(f"位移: {positive_envelope[i, 0]:.2f}, 力: {positive_envelope[i, 1]:.2f}")

    print("\n负向凸包络线关键点:")
    if 'negative_envelope' in locals():
        for i in range(len(negative_envelope)):
            print(f"位移: {negative_envelope[i, 0]:.2f}, 力: {negative_envelope[i, 1]:.2f}")
    save_envelope_data(input_file_path, positive_envelope if 'positive_envelope' in locals() else np.array([]),
                       negative_envelope if 'negative_envelope' in locals() else np.array([]))
if __name__ == "__main__":
    main_process()