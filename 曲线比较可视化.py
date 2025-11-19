# visualize_hysteresis_and_skeleton.py
import numpy as np
import matplotlib.pyplot as plt
import os


def read_hysteresis_data(file_path):
    """
    读取滞回曲线数据
    """
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 跳过标题行
            if line.startswith('"') or 'test' in line.lower():
                continue
            # 解析数据行
            try:
                values = line.split()
                if len(values) >= 2:
                    x = float(values[0])
                    y = float(values[1])
                    data.append([x, y])
            except ValueError:
                continue

    return np.array(data)


def read_skeleton_curve(file_path):
    """
    读取骨架曲线数据
    """
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 跳过标题行
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                values = line.split('\t')
                if len(values) >= 2:
                    x = float(values[0])
                    y = float(values[1])
                    data.append([x, y])
            except ValueError:
                continue

    return np.array(data)


def visualize_comparison(hysteresis_data, skeleton_data, title="Hysteresis Curve vs Skeleton Curve"):
    """
    可视化原始滞回曲线和骨架曲线对比
    """
    plt.figure(figsize=(12, 8))

    # 绘制原始滞回曲线（使用较小的点和透明度）
    plt.scatter(hysteresis_data[:, 0], hysteresis_data[:, 1],
                s=1, alpha=0.5, color='blue', label='Hysteresis Curve', zorder=1)

    # 绘制骨架曲线（使用较粗的线和明显颜色）
    if len(skeleton_data) > 0:
        plt.plot(skeleton_data[:, 0], skeleton_data[:, 1],
                 'r-', linewidth=2, marker='o', markersize=4,
                 label='Skeleton Curve', zorder=2)

    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加图例说明
    plt.text(0.02, 0.98, f'Hysteresis points: {len(hysteresis_data)}\nSkeleton points: {len(skeleton_data)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 可视化指定的滞回曲线和骨架曲线
    """
    # 文件路径配置
    hysteresis_file = r"F:\python学习\数据二\数据1.txt"
    skeleton_file = r"F:\python学习\数据二\数据1.txt"

    # 检查文件是否存在
    if not os.path.exists(hysteresis_file):
        print(f"错误：找不到滞回曲线文件 {hysteresis_file}")
        return

    if not os.path.exists(skeleton_file):
        print(f"错误：找不到骨架曲线文件 {skeleton_file}")
        return

    # 读取滞回曲线数据
    print("正在读取滞回曲线数据...")
    hysteresis_data = read_hysteresis_data(hysteresis_file)

    if len(hysteresis_data) == 0:
        print("错误：滞回曲线文件中没有有效数据")
        return

    print(f"成功读取 {len(hysteresis_data)} 个滞回曲线数据点")

    # 读取骨架曲线数据
    print("正在读取骨架曲线数据...")
    skeleton_data = read_skeleton_curve(skeleton_file)

    if len(skeleton_data) == 0:
        print("错误：骨架曲线文件中没有有效数据")
        return

    print(f"成功读取 {len(skeleton_data)} 个骨架曲线数据点")

    # 生成可视化
    print("正在生成可视化对比图...")
    visualize_comparison(hysteresis_data, skeleton_data,
                         "Hysteresis Curve vs Skeleton Curve - Test 806040")

    print("可视化完成！")


if __name__ == "__main__":
    main()
