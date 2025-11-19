import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os

class SkeletonCurveExtractor:
    """
    混合包络线法骨架曲线提取器（基于位移差分法分割半循环）
    """

    def extract_skeleton_curve(self, displacement: List[float], force: List[float]) -> Tuple[List[float], List[float]]:
        """
        使用新思路提取骨架曲线：第一圈循环的所有点 + 其他循环的峰值点

        Args:
            displacement: 位移数据列表
            force: 力数据列表

        Returns:
            骨架曲线的位移和力数据元组
        """
        # 识别滞回环（基于位移差分法分割半循环）
        cycles_data = self._identify_hysteresis_cycles_by_diff(displacement, force)

        if not cycles_data:
            return [], []

        # 使用新思路提取骨架曲线点
        skeleton_points = self._extract_skeleton_new_approach(cycles_data)

        # 移除了跳跃异常点检测步骤
        # filtered_points = self._remove_jump_outliers(skeleton_points)

        # 直接使用未过滤的骨架点
        final_points = skeleton_points

        # 分离位移和力数据
        skeleton_displacement = [point[0] for point in final_points]
        skeleton_force = [point[1] for point in final_points]

        return skeleton_displacement, skeleton_force

    def _identify_hysteresis_cycles_by_diff(self, displacement: List[float], force: List[float]) -> List[
        List[Tuple[float, float]]]:
        """
        基于位移本身正负转变识别滞回环（分割半循环）
        调整为：位移从由负变正开始到由正变负结束为一个半循环，
        或从由正变负开始到由负变正结束为一个半循环

        Args:
            displacement: 位移数据列表
            force: 力数据列表

        Returns:
            滞回环数据列表（每个完整循环由两个半循环组成）
        """
        if len(displacement) != len(force):
            raise ValueError("位移和力数据长度不匹配")

        if len(displacement) < 3:
            return [list(zip(displacement, force))]

        # 查找位移正负转变的点
        sign_change_indices = [0]  # 第一个点作为起始点

        for i in range(1, len(displacement)):
            # 如果前后位移符号不同，说明发生了正负转变
            if displacement[i - 1] * displacement[i] < 0:
                sign_change_indices.append(i)

        sign_change_indices.append(len(displacement) - 1)  # 最后一个点

        # 根据位移正负转变点分割半循环
        half_cycles = []
        for i in range(len(sign_change_indices) - 1):
            start_idx = sign_change_indices[i]
            end_idx = sign_change_indices[i + 1] + 1  # 包含结束点
            half_cycle_data = list(zip(displacement[start_idx:end_idx], force[start_idx:end_idx]))
            if len(half_cycle_data) > 1:  # 至少有两个点才构成半循环
                half_cycles.append(half_cycle_data)

        # 将相邻的两个半循环组合成完整循环
        full_cycles = []
        for i in range(0, len(half_cycles), 2):
            if i + 1 < len(half_cycles):
                # 合并两个半循环为一个完整循环
                combined_cycle = half_cycles[i] + half_cycles[i + 1][1:]  # 避免重复点
                full_cycles.append(combined_cycle)
            else:
                # 如果只有半个循环，单独作为一个循环
                full_cycles.append(half_cycles[i])

        return full_cycles

    def _extract_skeleton_new_approach(self, cycles_data: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
        """
        新思路：使用第一圈循环的所有点 + 其他循环的峰值点构成骨架曲线
        修改：最后一圈中如果峰值点是结束点，则舍弃

        Args:
            cycles_data: 多次循环加载的数据

        Returns:
            骨架曲线点列表
        """
        if not cycles_data:
            return []

        all_skeleton_points = []

        # 1. 添加第一圈循环的所有点
        first_cycle = cycles_data[0]
        all_skeleton_points.extend(first_cycle)

        # 2. 添加其他循环的峰值点（除了最后一圈的结束点）
        for i in range(1, len(cycles_data)):
            cycle = cycles_data[i]
            if not cycle:
                continue

            # 获取该循环的正向峰值点和负向峰值点
            positive_peak = max(cycle, key=lambda x: x[1])
            negative_peak = min(cycle, key=lambda x: x[1])

            # 如果是最后一圈，检查峰值点是否为结束点
            if i == len(cycles_data) - 1:  # 最后一圈
                last_point = cycle[-1]
                # 如果峰值点不是结束点才添加
                if positive_peak != last_point:
                    all_skeleton_points.append(positive_peak)
                if negative_peak != last_point:
                    all_skeleton_points.append(negative_peak)
            else:
                # 非最后一圈正常添加峰值点
                all_skeleton_points.append(positive_peak)
                all_skeleton_points.append(negative_peak)

        # 3. 按位移排序
        sorted_points = sorted(all_skeleton_points, key=lambda x: x[0])

        # 4. 处理重叠点
        final_curve = self._process_overlapping_points(sorted_points)

        return final_curve

    def _process_overlapping_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        处理重叠点，避免曲线回折

        Args:
            points: 排序后的所有点

        Returns:
            处理后的点列表
        """
        if not points:
            return []

        processed_points = [points[0]]

        for i in range(1, len(points)):
            current_point = points[i]
            previous_point = processed_points[-1]

            # 如果位移相同，取力绝对值最大的点
            if current_point[0] == previous_point[0]:
                if abs(current_point[1]) > abs(previous_point[1]):
                    processed_points[-1] = current_point
            else:
                processed_points.append(current_point)

        return processed_points

def read_txt_data(file_path: str) -> Tuple[List[float], List[float]]:
    """
    读取txt文件中的数据

    Args:
        file_path: txt文件路径

    Returns:
        位移和力数据列表
    """
    displacements = []
    forces = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 跳过第一行标题行
    for line in lines[1:]:
        line = line.strip()
        if line and not line.startswith('"'):  # 跳过空行和标题行
            try:
                values = line.split()
                if len(values) >= 2:
                    displacement = float(values[0])
                    force = float(values[1])
                    displacements.append(displacement)
                    forces.append(force)
            except ValueError:
                # 跳过无法解析的行
                continue

    return displacements, forces


def save_skeleton_data(displacement: List[float], force: List[float], original_file_path: str, save_dir: str):
    """
    保存骨架曲线数据到txt文件

    Args:
        displacement: 骨架曲线位移数据
        force: 骨架曲线力数据
        original_file_path: 原始文件路径
        save_dir: 保存目录
    """
    # 获取原始文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(original_file_path))[0]

    # 构造保存文件名
    save_file_name = f"{file_name}原始骨架曲线.txt"
    save_path = os.path.join(save_dir, save_file_name)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存数据
    with open(save_path, 'w') as file:
        file.write("Displacement\tForce\n")  # 写入标题行
        for disp, forc in zip(displacement, force):
            file.write(f"{disp}\t{forc}\n")

    print(f"骨架曲线数据已保存至: {save_path}")

def main_process():
    # 读取txt文件
    file_path = r'滞回曲线原始数据/umecus.txt'
    displacement, force = read_txt_data(file_path)

    # 创建骨架曲线提取器
    extractor = SkeletonCurveExtractor()

    # 提取骨架曲线
    skeleton_displacement, skeleton_force = extractor.extract_skeleton_curve(displacement, force)

    # 保存骨架曲线数据
    save_directory = r'F:\课题\计算机视觉识别屈服点项目\混合包络线法提取骨架曲线\骨架曲线原始数据'
    save_skeleton_data(skeleton_displacement, skeleton_force, file_path, save_directory)

    # 绘制对比图
    plt.figure(figsize=(12, 8))

    # 绘制原始滞回曲线
    plt.subplot(2, 1, 1)
    plt.plot(displacement, force, linewidth=1.0, alpha=0.7, color='blue', label='Hysteresis Curve')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title('Original Hysteresis Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 绘制骨架曲线
    plt.subplot(2, 1, 2)
    plt.plot(skeleton_displacement, skeleton_force, linewidth=2.0, color='red', marker='o',
             markersize=4, label='Skeleton Curve')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title('Extracted Skeleton Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 同时显示对比图
    plt.figure(figsize=(10, 6))
    plt.plot(displacement, force, linewidth=1.0, alpha=0.7, color='blue', label='Hysteresis Curve')
    plt.plot(skeleton_displacement, skeleton_force, linewidth=2.0, color='red', marker='o',
             markersize=4, label='Skeleton Curve')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title('Hysteresis Curve vs Skeleton Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main_process()