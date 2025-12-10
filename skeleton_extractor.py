import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
import csv
from io import StringIO
import os

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

    def calculate_envelope(self, skeleton_displacement, skeleton_force):
        """
        计算骨架曲线的外包络线

        Args:
            skeleton_displacement: 骨架曲线位移数据
            skeleton_force: 骨架曲线力数据

        Returns:
            正向外包络线和负向外包络线
        """
        skeleton_disp_array = np.array(skeleton_displacement)
        skeleton_force_array = np.array(skeleton_force)

        positive_indices = skeleton_disp_array >= 0
        negative_indices = skeleton_disp_array <= 0

        positive_points = np.column_stack((
            skeleton_disp_array[positive_indices],
            skeleton_force_array[positive_indices]
        ))
        negative_points = np.column_stack((
            skeleton_disp_array[negative_indices],
            skeleton_force_array[negative_indices]
        ))

        # 计算包络线
        positive_envelope = np.array([])
        negative_envelope = np.array([])

        if len(positive_points) > 2:
            positive_envelope = improved_geometric_filter(positive_points, 'positive')

        if len(negative_points) > 2:
            negative_envelope = improved_geometric_filter(negative_points, 'negative')
            
        return positive_envelope, negative_envelope


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


def read_txt_data(file_content, filename=''):
    """
    读取txt文件或csv文件中的数据
    
    Args:
        file_content: 文件内容（字符串或文件对象）
        filename: 文件名（用于判断是否为CSV文件）

    Returns:
        位移和力数据的numpy数组
    """
    try:
        # 根据文件名确定文件类型
        is_csv = filename.lower().endswith('.csv') if filename else False
        
        # 如果是上传的文件对象
        if hasattr(file_content, 'read'):
            content = file_content.read().decode('utf-8')
        else:
            content = file_content
            
        # 解析数据
        displacements = []
        forces = []
        if is_csv:
            # 处理CSV文件（跳过标题行）
            reader = csv.reader(StringIO(content))
            # 跳过标题行
            next(reader, None)

            for row in reader:
                if len(row) >= 2:
                    try:
                        displacement = float(row[0])
                        force = float(row[1])
                        displacements.append(displacement)
                        forces.append(force)
                    except ValueError:
                        # Skip rows that can't be parsed
                        continue
        else:
            lines = content.split('\n')

            for line in lines:
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

        return np.array(displacements), np.array(forces)
    except Exception as e:
        raise Exception(f"数据读取失败: {str(e)}")


def process_single_file(displacement, force, smooth_factor=0.05, num_points=300):
    """
    处理单个文件的所有步骤
    
    Args:
        displacement: 位移数据
        force: 力数据
        smooth_factor: RBF平滑因子
        num_points: 插值点数

    Returns:
        包含所有处理结果的字典
    """
    # 1. 提取骨架曲线
    extractor = SkeletonCurveExtractor()
    skeleton_displacement, skeleton_force = extractor.extract_skeleton_curve(
        displacement.tolist(), force.tolist()
    )

    # 2. 计算外包络线
    positive_envelope, negative_envelope = extractor.calculate_envelope(skeleton_displacement, skeleton_force)

    # 3. 合并外包络线数据
    all_envelope_points = []
    if len(positive_envelope) > 0:
        for point in positive_envelope:
            all_envelope_points.append([point[0], point[1]])

    if len(negative_envelope) > 0:
        for point in negative_envelope:
            all_envelope_points.append([point[0], point[1]])

    # 转换为numpy数组并排序
    if len(all_envelope_points) > 0:
        all_envelope_points = np.array(all_envelope_points)
        sorted_indices = np.argsort(all_envelope_points[:, 0])
        envelope_displacement = all_envelope_points[sorted_indices][:, 0]
        envelope_force = all_envelope_points[sorted_indices][:, 1]
    else:
        envelope_displacement = np.array([])
        envelope_force = np.array([])

    # 4. RBF插值平滑
    if len(envelope_displacement) > 0:
        x_smooth, y_smooth = rbf_smooth(
            envelope_displacement, envelope_force,
            function='multiquadric',
            smooth_factor=smooth_factor,
            num_points=num_points
        )
    else:
        x_smooth, y_smooth = np.array([]), np.array([])

    return {
        'skeleton_displacement': skeleton_displacement,
        'skeleton_force': skeleton_force,
        'positive_envelope': positive_envelope,
        'negative_envelope': negative_envelope,
        'envelope_displacement': envelope_displacement,
        'envelope_force': envelope_force,
        'x_smooth': x_smooth,
        'y_smooth': y_smooth
    }


def batch_process_files(file_paths, output_folder, extract_peak_points=True, 
                       extract_envelope=True, smooth_processing=True, 
                       smooth_factor=0.05, num_points=300):
    """
    批量处理文件
    
    Args:
        file_paths: 文件路径列表或上传的文件对象列表
        output_folder: 输出文件夹路径 (在Web应用中未使用)
        extract_peak_points: 是否提取骨架曲线峰值点
        extract_envelope: 是否提取外包络线
        smooth_processing: 是否进行平滑处理
        smooth_factor: RBF平滑因子
        num_points: 插值点数

    Returns:
        成功处理的文件数和失败文件列表
    """
    success_count = 0
    failed_files = []
    
    extractor = SkeletonCurveExtractor()
    
    for file_item in file_paths:
        try:
            # 获取文件名（根据不同类型的文件对象处理）
            if hasattr(file_item, 'name'):
                # Streamlit UploadedFile 对象
                file_name = file_item.name
                file_item.seek(0)
                file_content = file_item.read().decode('utf-8')
                file_item.seek(0)
            elif isinstance(file_item, str) and os.path.exists(file_item):
                # 本地文件路径
                file_name = os.path.basename(file_item)
                with open(file_item, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            else:
                # 其他情况
                failed_files.append(("未知文件", "无效的文件对象"))
                continue
                
            # 获取不带扩展名的文件名
            file_base_name = os.path.splitext(file_name)[0]
            
            # 判断文件类型
            is_csv = file_name.lower().endswith('.csv')
            
            # 解析数据
            displacements = []
            forces = []
            
            if is_csv:
                reader = csv.reader(StringIO(file_content))
                next(reader, None)  # 跳过标题行
                
                for row in reader:
                    if len(row) >= 2:
                        try:
                            displacement = float(row[0])
                            force = float(row[1])
                            displacements.append(displacement)
                            forces.append(force)
                        except ValueError:
                            continue
            else:
                lines = file_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('"'):
                        try:
                            values = line.split()
                            if len(values) >= 2:
                                displacement = float(values[0])
                                force = float(values[1])
                                displacements.append(displacement)
                                forces.append(force)
                        except ValueError:
                            continue
                            
            displacement = np.array(displacements)
            force = np.array(forces)
            
            if len(displacement) == 0 or len(force) == 0:
                failed_files.append((file_base_name, "数据为空"))
                continue
                
            # 提取骨架曲线
            skeleton_displacement, skeleton_force = extractor.extract_skeleton_curve(
                displacement.tolist(), force.tolist()
            )
            
            if len(skeleton_displacement) == 0 or len(skeleton_force) == 0:
                failed_files.append((file_base_name, "骨架曲线提取失败"))
                continue
            
            # 只有当所有勾选的操作都完成后才保存最终数据
            if extract_peak_points or extract_envelope or smooth_processing:
                # 准备最终数据
                final_data_columns = []
                final_data_values = []
                
                # 添加骨架曲线数据
                if extract_peak_points:
                    final_data_columns.extend(['骨架曲线_位移', '骨架曲线_力'])
                    final_data_values.extend([skeleton_displacement, skeleton_force])
                
                # 处理外包络线和平滑处理选项
                if extract_envelope or smooth_processing:
                    # 计算外包络线
                    positive_envelope, negative_envelope = extractor.calculate_envelope(skeleton_displacement, skeleton_force)
                    
                    # 合并外包络线数据
                    all_envelope_points = []
                    if len(positive_envelope) > 0:
                        for point in positive_envelope:
                            all_envelope_points.append([point[0], point[1]])

                    if len(negative_envelope) > 0:
                        for point in negative_envelope:
                            all_envelope_points.append([point[0], point[1]])

                    # 转换为numpy数组并排序
                    if len(all_envelope_points) > 0:
                        all_envelope_points = np.array(all_envelope_points)
                        sorted_indices = np.argsort(all_envelope_points[:, 0])
                        envelope_displacement = all_envelope_points[sorted_indices][:, 0]
                        envelope_force = all_envelope_points[sorted_indices][:, 1]
                    else:
                        envelope_displacement = np.array([])
                        envelope_force = np.array([])
                        
                    # 添加外包络线数据
                    if extract_envelope and len(envelope_displacement) > 0:
                        final_data_columns.extend(['外包络线_位移', '外包络线_力'])
                        final_data_values.extend([envelope_displacement.tolist(), envelope_force.tolist()])
                        
                    # 处理平滑数据
                    if smooth_processing and len(envelope_displacement) > 0:
                        # RBF插值平滑（使用用户设置的参数）
                        x_smooth, y_smooth = rbf_smooth(
                            envelope_displacement, envelope_force,
                            function='multiquadric',
                            smooth_factor=smooth_factor,  # 使用用户设置的参数
                            num_points=num_points         # 使用用户设置的参数
                        )
                        
                        final_data_columns.extend(['平滑包络线_位移', '平滑包络线_力'])
                        final_data_values.extend([x_smooth.tolist(), y_smooth.tolist()])
                
                # 保存最终数据（只保存一次）
                if final_data_columns:
                    # 确定最大长度以便对齐数据
                    max_length = max(len(arr) for arr in final_data_values)
                    
                    # 创建对齐的数据
                    aligned_data = []
                    for arr in final_data_values:
                        if len(arr) < max_length:
                            # 用NaN填充较短的数组
                            padded_arr = arr + [np.nan] * (max_length - len(arr))
                            aligned_data.append(padded_arr)
                        else:
                            aligned_data.append(arr)
                    
                    # 构建DataFrame
                    final_df_dict = {}
                    for i, column_name in enumerate(final_data_columns):
                        final_df_dict[column_name] = aligned_data[i]
                    
                    import pandas as pd
                    final_df = pd.DataFrame(final_df_dict)
                    
                    # 在Web应用中，我们将结果保存到session_state而不是文件
                    if output_folder is None:
                        # Web应用模式 - 结果保存在内存中供后续下载
                        if 'batch_results' not in st.session_state:
                            st.session_state.batch_results = {}
                        st.session_state.batch_results[file_base_name] = final_df
                    else:
                        # 本地模式 - 保存到文件
                        final_output_path = os.path.join(output_folder, f"{file_base_name}_骨架曲线数据.csv")
                        final_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
                    
            success_count += 1
            
        except Exception as e:
            failed_files.append((file_base_name if 'file_base_name' in locals() else "未知文件", str(e)))
            
    return success_count, failed_files
