# visualize_single_skeleton.py
import numpy as np
import matplotlib.pyplot as plt
import os


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


def visualize_skeleton(skeleton_data, title="Skeleton Curve"):
    """
    可视化骨架曲线
    """
    plt.figure(figsize=(12, 8))

    # 绘制骨架曲线（使用较粗的线和明显颜色）
    if len(skeleton_data) > 0:
        plt.plot(skeleton_data[:, 0], skeleton_data[:, 1],
                 'r-', linewidth=2, marker='o', markersize=4,
                 label='Skeleton Curve')

    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加数据点统计信息
    plt.text(0.02, 0.98, f'Skeleton points: {len(skeleton_data)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 可视化指定的骨架曲线
    """
    # 文件路径配置
    skeleton_file = r"F:\python学习\promax_skeleton_curves\kow01no4_curves_smooth_plus.txt"

    # 检查文件是否存在
    if not os.path.exists(skeleton_file):
        print(f"错误：找不到骨架曲线文件 {skeleton_file}")
        return

    # 读取骨架曲线数据
    print("正在读取骨架曲线数据...")
    skeleton_data = read_skeleton_curve(skeleton_file)

    if len(skeleton_data) == 0:
        print("错误：骨架曲线文件中没有有效数据")
        return

    print(f"成功读取 {len(skeleton_data)} 个骨架曲线数据点")

    # 生成可视化
    print("正在生成可视化图表...")
    visualize_skeleton(skeleton_data, "Skeleton Curve - Test 806040")

    print("可视化完成！")


if __name__ == "__main__":
    main()
