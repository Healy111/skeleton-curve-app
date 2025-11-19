# skeleton_curve_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from io import StringIO
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 设置页面配置
st.set_page_config(
    page_title="骨架曲线提取器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和介绍
st.title("📈 骨架曲线提取器")
st.markdown("---")

# 在侧边栏添加使用说明
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. **上传数据文件**：选择包含位移和力数据的.txt或.csv文件
    2. **调整参数**（可选）：根据需要调整处理参数
    3. **提取骨架曲线**：点击按钮开始处理
    4. **查看结果**：分析生成的图表和下载处理结果

    **数据格式要求：**
    - 文本文件(.txt格式)或CSV文件(.csv格式)
    - 包含两列数据：位移和力
    - TXT文件：列之间用空格或制表符分隔
    - CSV文件：包含表头行，从第二行开始为数据
    """)

# 文件上传区域
st.header("📁 文件上传")
uploaded_file = st.file_uploader(
    "选择滞回曲线数据文件",
    type=['txt','csv'],
    help="请上传包含位移和力两列数据的文本文件或CSV文件"
)

# 处理参数设置
st.header("⚙️ 处理参数")
col1, col2 = st.columns(2)

with col1:
    smooth_factor = st.slider(
        "RBF平滑因子",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="控制RBF插值的平滑程度"
    )

with col2:
    num_points = st.slider(
        "插值点数",
        min_value=100,
        max_value=500,
        value=300,
        step=50,
        help="平滑曲线上的点数"
    )


def read_txt_data(file_content):
    """读取txt文件或csv文件中的数据"""
    try:
        # 根据文件名确定文件类型
        filename = getattr(file_content, 'name', '')
        is_csv = filename.lower().endswith('.csv')
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
            lines = content.split('\n')
            # 跳过第一行
            data_lines = lines[1:] if len(lines) > 1 else []


            reader = csv.reader(StringIO(content))
            # Skip header row
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
        st.error(f"数据读取失败: {str(e)}")
        return None, None


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



# 处理按钮
if st.button("🚀 提取骨架曲线", type="primary", use_container_width=True):
    if uploaded_file is not None:
        # 检查文件格式
        filename = uploaded_file.name
        if not (filename.endswith('.txt') or filename.endswith('.csv')):
            st.error("❌ 不支持的文件格式，请上传 .txt 或 .csv 文件")
            st.stop()
        try:
            with st.spinner("正在处理数据，请稍候..."):

                # 1. 读取数据
                displacement, force = read_txt_data(uploaded_file)

                if displacement is None or force is None:
                    st.error("数据读取失败，请检查文件格式")
                    st.stop()

                # 2. 提取骨架曲线
                extractor = SkeletonCurveExtractor()
                skeleton_displacement, skeleton_force = extractor.extract_skeleton_curve(
                    displacement.tolist(), force.tolist()
                )

                # 3. 计算外包络线
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

            # 显示成功信息
            st.success("✅ 处理完成！")

            # 5. 显示结果图表
            st.header("📊 处理结果")

            # 创建四个子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 原始滞回曲线
            axes[0, 0].plot(displacement, force, linewidth=1.0, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('位移')
            axes[0, 0].set_ylabel('力')
            axes[0, 0].set_title('原始滞回曲线')
            axes[0, 0].grid(True, alpha=0.3)

            # 骨架曲线
            axes[0, 1].plot(skeleton_displacement, skeleton_force, linewidth=2.0,
                            color='red', marker='o', markersize=4)
            axes[0, 1].set_xlabel('位移')
            axes[0, 1].set_ylabel('力')
            axes[0, 1].set_title('提取的骨架曲线')
            axes[0, 1].grid(True, alpha=0.3)

            # 外包络线
            axes[1, 0].plot(skeleton_displacement, skeleton_force, linewidth=1.0,
                            alpha=0.7, color='blue', label='骨架曲线')
            if len(positive_envelope) > 0:
                axes[1, 0].plot(positive_envelope[:, 0], positive_envelope[:, 1],
                                'r-', linewidth=2.5, label='正向包络线')
            if len(negative_envelope) > 0:
                axes[1, 0].plot(negative_envelope[:, 0], negative_envelope[:, 1],
                                'g-', linewidth=2.5, label='负向包络线')
            axes[1, 0].set_xlabel('位移')
            axes[1, 0].set_ylabel('力')
            axes[1, 0].set_title('骨架曲线与外包络线')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # RBF平滑结果
            if len(envelope_displacement) > 0:
                axes[1, 1].scatter(envelope_displacement, envelope_force, c='blue',
                                   s=30, label='原始包络点', zorder=3)
            if len(x_smooth) > 0:
                axes[1, 1].plot(x_smooth, y_smooth, 'r-', linewidth=2.5,
                                label='RBF平滑包络线')
            axes[1, 1].set_xlabel('位移')
            axes[1, 1].set_ylabel('力')
            axes[1, 1].set_title('RBF插值平滑处理')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # 数据下载区域
            st.header("💾 下载结果")

            col1, col2 = st.columns(2)

            with col1:
                # 下载骨架曲线数据
                if len(skeleton_displacement) > 0:
                    skeleton_df = pd.DataFrame({
                        '位移': skeleton_displacement,
                        '力': skeleton_force
                    })
                    skeleton_csv = skeleton_df.to_csv(index=False)
                    st.download_button(
                        label="下载骨架曲线数据",
                        data=skeleton_csv,
                        file_name="skeleton_curve.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with col2:
                # 下载平滑包络线数据
                if len(x_smooth) > 0:
                    envelope_df = pd.DataFrame({
                        '位移': x_smooth,
                        '力': y_smooth
                    })
                    envelope_csv = envelope_df.to_csv(index=False)
                    st.download_button(
                        label="下载平滑包络线数据",
                        data=envelope_csv,
                        file_name="smoothed_envelope.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {str(e)}")
            st.info("请检查数据文件格式是否正确")
    else:
        st.warning("⚠️ 请先上传数据文件")

# 数据显示区域（如果用户想查看原始数据）
if uploaded_file is not None:
    with st.expander("📋 查看原始数据"):
        try:
            # 重置文件指针
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf-8')

            # 解析并显示数据
            displacements = []
            forces = []
            lines = content.split('\n')

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

            data_preview = pd.DataFrame({
                '位移': displacements[:10],
                '力': forces[:10]
            })
            st.dataframe(data_preview, use_container_width=True)
            st.write(f"数据总行数: {len(displacements)}")
        except Exception as e:
            st.error(f"数据显示错误: {str(e)}")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>骨架曲线提取器 Web版</div>",
    unsafe_allow_html=True
)
