# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import requests
import os
import matplotlib.font_manager as fm
import csv
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

# Import core functionality from the backend module
from skeleton_extractor import (
    SkeletonCurveExtractor,
    improved_geometric_filter,
    rbf_smooth
)

def setup_chinese_font_for_matplotlib():
    """专门为matplotlib设置中文字体"""
    try:
        # 方案1：下载思源黑体
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        font_path = os.path.join(tempfile.gettempdir(), "NotoSansCJKsc.otf")
        
        if not os.path.exists(font_path):
            response = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(response.content)
        
        # 注册字体
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        # 方案2：使用系统字体
        try:
            system_fonts = ['DejaVu Sans', 'Arial']
            plt.rcParams['font.family'] = system_fonts
            plt.rcParams['axes.unicode_minus'] = False
            return False
        except:
            return False

# 在应用开头调用字体设置
setup_chinese_font_for_matplotlib()
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
    type=['txt', 'csv'],
    help="请上传包含位移和力两列数据的文本文件或CSV文件"
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
            # 重置文件指针，以便后续再次读取
        if hasattr(file_content, 'seek'):
            file_content.seek(0)

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


# 数据显示区域（如果用户想查看原始数据）
if uploaded_file is not None:
    with st.expander("📋 查看原始数据"):
        try:
            # 重置文件指针
            uploaded_file.seek(0)
            displacement_data, force_data = read_txt_data(uploaded_file)
            if displacement_data is not None and force_data is not None:
                data_preview = pd.DataFrame({
                    '位移': displacement_data[:10],
                    '力': force_data[:10]
                })
                st.dataframe(data_preview, use_container_width=True)
                st.write(f"数据总行数: {len(displacement_data)}")
        except Exception as e:
            st.error(f"数据显示错误: {str(e)}")

# 处理参数设置
st.header("⚙️ 处理参数")
# 在处理按钮之后添加新功能按钮
colC1, colC2 = st.columns(2)

with colC1:
    clear_data = st.button("🗑️ 清空数据", use_container_width=True)
    # 处理清空数据功能
    if clear_data:
        if 'processed' in st.session_state:
            del st.session_state['processed']
        if 'results' in st.session_state:
            del st.session_state['results']
        st.rerun()

with colC2:
    reset_params = st.button("↺ 重置参数", use_container_width=True)
    # 处理参数重置功能
    if reset_params:
        st.session_state['smooth_factor'] = 0.05
        st.session_state['num_points'] = 300
        st.rerun()

colA1, colA2 = st.columns(2)

with colA1:
    # 从 session state 获取值，如果不存在则使用默认值
    if 'smooth_factor' not in st.session_state:
        st.session_state['smooth_factor'] = 0.05
    smooth_factor = st.slider(
        "RBF平滑因子",
        min_value=0.01,
        max_value=0.2,
        value=st.session_state['smooth_factor'],  # 从 session state 读取值
        step=0.01,
        help="控制RBF插值的平滑程度"
    )
    # 更新 session state
    st.session_state['smooth_factor'] = smooth_factor

with colA2:
    # 从 session state 获取值，如果不存在则使用默认值
    if 'num_points' not in st.session_state:
        st.session_state['num_points'] = 300
    num_points = st.slider(
        "插值点数",
        min_value=100,
        max_value=500,
        value=st.session_state['num_points'],  # 从 session state 读取值
        step=50,
        help="平滑曲线上的点数"
    )
    # 更新 session state
    st.session_state['num_points'] = num_points

if st.button("🚀 提取骨架曲线", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ 请先上传数据文件")
        st.stop()

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

        # 保存结果到session_state
        st.session_state['processed'] = True
        st.session_state['results'] = {
            'displacement': displacement,
            'force': force,
            'skeleton_displacement': skeleton_displacement,
            'skeleton_force': skeleton_force,
            'positive_envelope': positive_envelope,
            'negative_envelope': negative_envelope,
            'envelope_displacement': envelope_displacement,
            'envelope_force': envelope_force,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth
        }

        # 显示成功信息
        st.success("✅ 处理完成！")

    except Exception as e:
        st.error(f"❌ 处理过程中发生错误: {str(e)}")
        st.info("请检查数据文件格式是否正确")

# 在按钮之外显示结果，确保重新运行时也能显示
if st.session_state.get('processed', False) and 'results' in st.session_state:
    results = st.session_state['results']

    # 显示结果图表
    st.header("📊 处理结果")

    # 创建四个子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 原始滞回曲线
    axes[0, 0].plot(results['displacement'], results['force'], linewidth=1.0, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('位移')
    axes[0, 0].set_ylabel('力')
    axes[0, 0].set_title('原始滞回曲线')
    axes[0, 0].grid(True, alpha=0.3)

    # 骨架曲线
    axes[0, 1].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=2.0,
                    color='red', marker='o', markersize=4)
    axes[0, 1].set_xlabel('位移')
    axes[0, 1].set_ylabel('力')
    axes[0, 1].set_title('提取的骨架曲线')
    axes[0, 1].grid(True, alpha=0.3)

    # 外包络线
    axes[1, 0].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=1.0,
                    alpha=0.7, color='blue', label='骨架曲线')
    if len(results['positive_envelope']) > 0:
        axes[1, 0].plot(results['positive_envelope'][:, 0], results['positive_envelope'][:, 1],
                        'r-', linewidth=2.5, label='正向包络线')
    if len(results['negative_envelope']) > 0:
        axes[1, 0].plot(results['negative_envelope'][:, 0], results['negative_envelope'][:, 1],
                        'g-', linewidth=2.5, label='负向包络线')
    axes[1, 0].set_xlabel('位移')
    axes[1, 0].set_ylabel('力')
    axes[1, 0].set_title('骨架曲线与外包络线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # RBF平滑结果
    if len(results['envelope_displacement']) > 0:
        axes[1, 1].scatter(results['envelope_displacement'], results['envelope_force'], c='blue',
                           s=30, label='原始包络点', zorder=3)
    if len(results['x_smooth']) > 0:
        axes[1, 1].plot(results['x_smooth'], results['y_smooth'], 'r-', linewidth=2.5,
                        label='RBF平滑包络线')
    axes[1, 1].set_xlabel('位移')
    axes[1, 1].set_ylabel('力')
    axes[1, 1].set_title('RBF插值平滑处理')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # 数据下载区域
    st.header("💾 数据操作")

    colB1, colB2, colB3 = st.columns(3)
    with colB1:
        # 下载骨架曲线数据
        if len(results['skeleton_displacement']) > 0:
            skeleton_df = pd.DataFrame({
                '位移': results['skeleton_displacement'],
                '力': results['skeleton_force']
            })
            skeleton_csv =  "\ufeff" + skeleton_df.to_csv(index=False)
            st.download_button(
                label="下载骨架曲线数据",
                data=skeleton_csv,
                file_name="skeleton_curve.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_skeleton"  # 添加唯一key
            )

    with colB2:
        # 下载平滑包络线数据
        if len(results['x_smooth']) > 0:
            envelope_df = pd.DataFrame({
                '位移': results['x_smooth'],
                '力': results['y_smooth']
            })
            envelope_csv = "\ufeff" +  envelope_df.to_csv(index=False)
            st.download_button(
                label="下载平滑包络线数据",
                data=envelope_csv,
                file_name="smoothed_envelope.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_envelope"  # 添加唯一key
            )

    # 图表下载功能
    st.header("📊 图表操作")

    # 创建图表下载
    img_buffer = BytesIO()
    plt.figure(figsize=(15, 12))

    # 复制上面的图表创建逻辑
    fig_download, axes_download = plt.subplots(2, 2, figsize=(15, 12))

    # 原始滞回曲线
    axes_download[0, 0].plot(results['displacement'], results['force'], linewidth=1.0, alpha=0.7, color='blue')
    axes_download[0, 0].set_xlabel('位移')
    axes_download[0, 0].set_ylabel('力')
    axes_download[0, 0].set_title('原始滞回曲线')
    axes_download[0, 0].grid(True, alpha=0.3)

    # 骨架曲线
    axes_download[0, 1].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=2.0,
                             color='red', marker='o', markersize=4)
    axes_download[0, 1].set_xlabel('位移')
    axes_download[0, 1].set_ylabel('力')
    axes_download[0, 1].set_title('提取的骨架曲线')
    axes_download[0, 1].grid(True, alpha=0.3)

    # 外包络线
    axes_download[1, 0].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=1.0,
                             alpha=0.7, color='blue', label='骨架曲线')
    if len(results['positive_envelope']) > 0:
        axes_download[1, 0].plot(results['positive_envelope'][:, 0], results['positive_envelope'][:, 1],
                                 'r-', linewidth=2.5, label='正向包络线')
    if len(results['negative_envelope']) > 0:
        axes_download[1, 0].plot(results['negative_envelope'][:, 0], results['negative_envelope'][:, 1],
                                 'g-', linewidth=2.5, label='负向包络线')
    axes_download[1, 0].set_xlabel('位移')
    axes_download[1, 0].set_ylabel('力')
    axes_download[1, 0].set_title('骨架曲线与外包络线')
    axes_download[1, 0].legend()
    axes_download[1, 0].grid(True, alpha=0.3)

    # RBF平滑结果
    if len(results['envelope_displacement']) > 0:
        axes_download[1, 1].scatter(results['envelope_displacement'], results['envelope_force'], c='blue',
                                    s=30, label='原始包络点', zorder=3)
    if len(results['x_smooth']) > 0:
        axes_download[1, 1].plot(results['x_smooth'], results['y_smooth'], 'r-', linewidth=2.5,
                                 label='RBF平滑包络线')
    axes_download[1, 1].set_xlabel('位移')
    axes_download[1, 1].set_ylabel('力')
    axes_download[1, 1].set_title('RBF插值平滑处理')
    axes_download[1, 1].legend()
    axes_download[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig_download)

    # 下载图表按钮
    st.download_button(
        label="📥 下载图表PNG",
        data=img_buffer,
        file_name="skeleton_curve_analysis.png",
        mime="image/png",
        use_container_width=True,
        key="download_chart"  # 添加唯一key
    )

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>骨架曲线提取器 Web版</div>",
    unsafe_allow_html=True
)
