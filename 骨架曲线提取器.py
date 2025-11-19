from 混合包络线法txt import *
from RBF插值 import rbf_smooth
from 外包络线 import improved_geometric_filter

plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def main_process():
    # 1. 读取原始数据
    file_path = r'滞回曲线原始数据/at75n11.txt'  # 请根据实际路径修改
    displacement, force = read_txt_data(file_path)

    # 2. 提取骨架曲线
    extractor = SkeletonCurveExtractor()
    skeleton_displacement, skeleton_force = extractor.extract_skeleton_curve(displacement, force)

    # 3. 计算外包络线
    # 分离正负位移数据
    skeleton_disp_array = np.array(skeleton_displacement)
    skeleton_force_array = np.array(skeleton_force)

    positive_indices = skeleton_disp_array >= 0
    negative_indices = skeleton_disp_array <= 0  # 包含0点

    positive_points = np.column_stack((skeleton_disp_array[positive_indices], skeleton_force_array[positive_indices]))
    negative_points = np.column_stack((skeleton_disp_array[negative_indices], skeleton_force_array[negative_indices]))

    # 计算包络线
    if len(positive_points) > 2:
        positive_envelope = improved_geometric_filter(positive_points, 'positive')

    if len(negative_points) > 2:
        negative_envelope = improved_geometric_filter(negative_points, 'negative')

    # 合并外包络线数据并排序
    all_envelope_points = []
    if 'positive_envelope' in locals() and len(positive_envelope) > 0:
        for point in positive_envelope:
            all_envelope_points.append([point[0], point[1]])

    if 'negative_envelope' in locals() and len(negative_envelope) > 0:
        for point in negative_envelope:
            all_envelope_points.append([point[0], point[1]])

    # 转换为numpy数组并按位移排序
    if len(all_envelope_points) > 0:
        all_envelope_points = np.array(all_envelope_points)
        sorted_indices = np.argsort(all_envelope_points[:, 0])
        envelope_displacement = all_envelope_points[sorted_indices][:, 0]
        envelope_force = all_envelope_points[sorted_indices][:, 1]

    # 4. RBF插值平滑
    if len(all_envelope_points) > 0:
        x_smooth, y_smooth = rbf_smooth(envelope_displacement, envelope_force,
                                        function='multiquadric', smooth_factor=0.05, num_points=300)

    # 5. 可视化所有步骤
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 原始滞回曲线
    axes[0, 0].plot(displacement, force, linewidth=1.0, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('位移')
    axes[0, 0].set_ylabel('力')
    axes[0, 0].set_title('原始滞回曲线')
    axes[0, 0].grid(True, alpha=0.3)

    # 骨架曲线
    axes[0, 1].plot(skeleton_displacement, skeleton_force, linewidth=2.0, color='red', marker='o', markersize=4)
    axes[0, 1].set_xlabel('位移')
    axes[0, 1].set_ylabel('力')
    axes[0, 1].set_title('提取的骨架曲线')
    axes[0, 1].grid(True, alpha=0.3)

    # 外包络线
    axes[1, 0].plot(skeleton_displacement, skeleton_force, linewidth=1.0, alpha=0.7, color='blue', label='骨架曲线')
    if 'positive_envelope' in locals() and len(positive_envelope) > 0:
        axes[1, 0].plot(positive_envelope[:, 0], positive_envelope[:, 1], 'r-', linewidth=2.5, label='正向包络线')
    if 'negative_envelope' in locals() and len(negative_envelope) > 0:
        axes[1, 0].plot(negative_envelope[:, 0], negative_envelope[:, 1], 'g-', linewidth=2.5, label='负向包络线')
    axes[1, 0].set_xlabel('位移')
    axes[1, 0].set_ylabel('力')
    axes[1, 0].set_title('骨架曲线与外包络线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # RBF平滑结果
    axes[1, 1].scatter(envelope_displacement, envelope_force, c='blue', s=30, label='原始包络点', zorder=3)
    if len(all_envelope_points) > 0:
        axes[1, 1].plot(x_smooth, y_smooth, 'r-', linewidth=2.5, label='RBF平滑包络线')
    axes[1, 1].set_xlabel('位移')
    axes[1, 1].set_ylabel('力')
    axes[1, 1].set_title('RBF插值平滑处理')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("处理完成！各步骤结果已可视化显示。")


# 运行主流程
if __name__ == "__main__":
    main_process()