# skeleton_extractor.py
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull

class SkeletonCurveExtractor:
    """
    混合包络线法骨架曲线提取器（基于位移差分法分割半循环）
    """

    def extract_skeleton_curve(self, displacement: list, force: list) -> tuple:
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

        # 分离位移和力数据
        skeleton_displacement = [point[0] for point in skeleton_points]
        skeleton_force = [point[1] for point in skeleton_points]

        return skeleton_displacement, skeleton_force

    def _identify_hysteresis_cycles_by_diff(self, displacement: list, force: list) -> list:
        """
        基于位移本身正负转变识别滞回环（分割半循环）

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

    def _extract_skeleton_new_approach(self, cycles_data: list) -> list:
        """
        新思路：使用第一圈循环的所有点 + 其他循环的峰值点构成骨架曲线

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

    def _process_overlapping_points(self, points: list) -> list:
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


def improved_geometric_filter(points, side='positive'):
    """改进的几何特性包络点筛选，考虑骨架曲线力绝对值先增后减的特点"""
    if len(points) < 3:
        return points

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except:
        # 如果凸包计算失败，直接返回输入点
        return points

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
    x_new = np.linspace(x.min(), x.max(), num_points)

     # 使用RBF进行插值
    rbf = Rbf(x, y, function=function, smooth=smooth_factor)
    y_smooth = rbf(x_new)
    return x_new, y_smooth
