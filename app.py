import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO
import glob

# 从后端模块导入核心功能
from skeleton_extractor import (
    read_txt_data,
    process_single_file,
    batch_process_files
)

# 设置页面配置
st.set_page_config(
    page_title="骨架曲线提取器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'home'

# 页面导航函数
def navigate_to(mode):
    st.session_state.mode = mode
    st.rerun()

# 主页 - 选择处理模式
if st.session_state.mode == 'home':
    st.title("📈 骨架曲线提取器")
    st.markdown("---")
    st.header("请选择处理模式")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔬 单独处理", type="primary", width='stretch'):
            navigate_to('single')
    
    with col2:
        if st.button("🏭 批量处理", type="primary", width='stretch'):
            navigate_to('batch')
            
    # 在侧边栏添加使用说明
    with st.sidebar:
        st.header("使用说明")
        st.markdown("""
        **两种处理模式：**

        1. **单独处理**：适用于单个文件的详细分析和可视化展示
        2. **批量处理**：适用于多个文件的一键批处理，无需可视化展示

        **通用数据格式要求：**
        - 文本文件(.txt格式)或CSV文件(.csv格式)
        - 包含两列数据：位移和力
        - TXT文件：列之间用空格或制表符分隔
        - CSV文件：包含表头行，从第二行开始为数据
        """)
        
    st.stop()

# 单独处理模式
elif st.session_state.mode == 'single':
    # 标题和介绍
    st.title("📈 骨架曲线提取器 - 单独处理模式")
    st.markdown("---")
    
    # 返回主页按钮
    if st.button("🏠 返回主页"):
        navigate_to('home')

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


    # 数据显示区域（如果用户想查看原始数据）
    if uploaded_file is not None:
        with st.expander("📋 查看原始数据"):
            try:
                # 重置文件指针
                uploaded_file.seek(0)
                displacement_data, force_data = read_txt_data(uploaded_file, uploaded_file.name)
                uploaded_file.seek(0)
                if displacement_data is not None and force_data is not None:
                    data_preview = pd.DataFrame({
                        '位移': displacement_data[:10],
                        '力': force_data[:10]
                    })
                    st.dataframe(data_preview, width='stretch')
                    st.write(f"数据总行数: {len(displacement_data)}")
            except Exception as e:
                st.error(f"数据显示错误: {str(e)}")

    # 处理参数设置
    st.header("⚙️ 处理参数")
    # 在处理按钮之后添加新功能按钮
    colC1, colC2 = st.columns(2)

    with colC1:
        clear_data = st.button("🗑️ 清空数据", width='stretch')
        # 处理清空数据功能
        if clear_data:
            if 'processed' in st.session_state:
                del st.session_state['processed']
            if 'results' in st.session_state:
                del st.session_state['results']
            st.rerun()

    with colC2:
        reset_params = st.button("↺ 重置参数", width='stretch')
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

    if st.button("🚀 提取骨架曲线", type="primary", width='stretch'):
        if uploaded_file is None:
            st.warning("⚠️ 请先上传数据文件")
            st.stop()

        # 检查文件格式
        filename = uploaded_file.name
        if not (filename.endswith('.txt') or filename.endswith('.csv')):
            st.error("❌ 不支持的文件格式，请上传 .txt 或 .csv 文件")
            st.stop()
        
        # 获取不带扩展名的文件名，用于下载文件命名
        file_base_name = os.path.splitext(filename)[0]

        try:
            with st.spinner("正在处理数据，请稍候..."):
                # 1. 读取数据
                uploaded_file.seek(0)
                displacement, force = read_txt_data(uploaded_file, filename)
                # 重置文件指针
                uploaded_file.seek(0)
                if displacement is None or force is None:
                    st.error("数据读取失败，请检查文件格式")
                    st.stop()

                # 2. 处理数据
                results = process_single_file(displacement, force, smooth_factor, num_points)

            # 保存结果到session_state，同时保存文件名
            st.session_state['processed'] = True
            st.session_state['results'] = {
                'displacement':displacement,
                'force':force,
                'skeleton_displacement': results['skeleton_displacement'],
                'skeleton_force': results['skeleton_force'],
                'positive_envelope': results['positive_envelope'],
                'negative_envelope': results['negative_envelope'],
                'envelope_displacement': results['envelope_displacement'],
                'envelope_force': results['envelope_force'],
                'x_smooth': results['x_smooth'],
                'y_smooth': results['y_smooth'],
                'file_base_name': file_base_name  # 保存文件名前缀
            }

            # 显示成功信息
            st.success("✅ 处理完成！")

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {str(e)}")
            st.info("请检查数据文件格式是否正确")

    # 在按钮之外显示结果，确保重新运行时也能显示
    if st.session_state.get('processed', False) and 'results' in st.session_state:
        results = st.session_state['results']
        # 获取文件名前缀
        file_base_name = results.get('file_base_name', 'data')

        # 显示结果图表
        st.header("📊 处理结果")

        # 创建四个子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 原始滞回曲线
        axes[0, 0].plot(results['displacement'], results['force'], linewidth=1.0, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('displacement')
        axes[0, 0].set_ylabel('force')
        axes[0, 0].set_title('Original Hysteresis Curve')
        axes[0, 0].grid(True, alpha=0.3)

        # 骨架曲线
        axes[0, 1].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=2.0,
                        color='red', marker='o', markersize=4)
        axes[0, 1].set_xlabel('displacement')
        axes[0, 1].set_ylabel('force')
        axes[0, 1].set_title('Skeleton Curve')
        axes[0, 1].grid(True, alpha=0.3)

        # 外包络线
        axes[1, 0].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=1.0,
                        alpha=0.7, color='blue', label='Skeleton Curve')
        if len(results['positive_envelope']) > 0:
            axes[1, 0].plot(results['positive_envelope'][:, 0], results['positive_envelope'][:, 1],
                            'r-', linewidth=2.5, label='Positive Envelope')
        if len(results['negative_envelope']) > 0:
            axes[1, 0].plot(results['negative_envelope'][:, 0], results['negative_envelope'][:, 1],
                            'g-', linewidth=2.5, label='Negative Envelope')
        axes[1, 0].set_xlabel('displacement')
        axes[1, 0].set_ylabel('force')
        axes[1, 0].set_title('Skeleton Curve and Envelope')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # RBF平滑结果
        if len(results['envelope_displacement']) > 0:
            axes[1, 1].scatter(results['envelope_displacement'], results['envelope_force'], c='blue',
                               s=30, label='Original Envelope Points', zorder=3)
        if len(results['x_smooth']) > 0:
            axes[1, 1].plot(results['x_smooth'], results['y_smooth'], 'r-', linewidth=2.5,
                            label='RBF Smoothed Envelope')
        axes[1, 1].set_xlabel('displacement')
        axes[1, 1].set_ylabel('force')
        axes[1, 1].set_title('RBF Smoothed Envelope')
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
                skeleton_csv ="\ufeff" + skeleton_df.to_csv(index=False)
                st.download_button(
                    label="下载骨架曲线数据",
                    data=skeleton_csv,
                    file_name=f"{file_base_name}_skeleton_curve.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_skeleton"  # 添加唯一key
                )

        with colB2:
            # 下载平滑包络线数据
            if len(results['x_smooth']) > 0:
                envelope_df = pd.DataFrame({
                    '位移': results['x_smooth'],
                    '力': results['y_smooth']
                })
                envelope_csv = "\ufeff" + envelope_df.to_csv(index=False)
                st.download_button(
                    label="下载平滑包络线数据",
                    data=envelope_csv,
                    file_name=f"{file_base_name}_smoothed_envelope.csv",
                    mime="text/csv",
                    width='stretch',
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
        axes_download[0, 0].set_xlabel('displacement')
        axes_download[0, 0].set_ylabel('force')
        axes_download[0, 0].set_title('Original Hysteresis Curve')
        axes_download[0, 0].grid(True, alpha=0.3)

        # 骨架曲线
        axes_download[0, 1].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=2.0,
                                 color='red', marker='o', markersize=4)
        axes_download[0, 1].set_xlabel('displacement')
        axes_download[0, 1].set_ylabel('force')
        axes_download[0, 1].set_title('Skeleton Curve')
        axes_download[0, 1].grid(True, alpha=0.3)

        # 外包络线
        axes_download[1, 0].plot(results['skeleton_displacement'], results['skeleton_force'], linewidth=1.0,
                                 alpha=0.7, color='blue', label='Skeleton Curve')
        if len(results['positive_envelope']) > 0:
            axes_download[1, 0].plot(results['positive_envelope'][:, 0], results['positive_envelope'][:, 1],
                                     'r-', linewidth=2.5, label='Positive Envelope')
        if len(results['negative_envelope']) > 0:
            axes_download[1, 0].plot(results['negative_envelope'][:, 0], results['negative_envelope'][:, 1],
                                     'g-', linewidth=2.5, label='Negative Envelope')
        axes_download[1, 0].set_xlabel('displacement')
        axes_download[1, 0].set_ylabel('force')
        axes_download[1, 0].set_title(' Skeleton Curve and Envelope')
        axes_download[1, 0].legend()
        axes_download[1, 0].grid(True, alpha=0.3)

        # RBF平滑结果
        if len(results['envelope_displacement']) > 0:
            axes_download[1, 1].scatter(results['envelope_displacement'], results['envelope_force'], c='blue',
                                        s=30, label='Original Envelope Points', zorder=3)
        if len(results['x_smooth']) > 0:
            axes_download[1, 1].plot(results['x_smooth'], results['y_smooth'], 'r-', linewidth=2.5,
                                     label='RBF Smoothed Envelope')
        axes_download[1, 1].set_xlabel('displacement')
        axes_download[1, 1].set_ylabel('force')
        axes_download[1, 1].set_title('RBF Smoothed Envelope')
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
            file_name=f"{file_base_name}_skeleton_curve_analysis.png",
            mime="image/png",
            width='stretch',
            key="download_chart"  # 添加唯一key
        )

    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>骨架曲线提取器 Web版 - 单独处理模式</div>",
        unsafe_allow_html=True
    )

# 批量处理模式
elif st.session_state.mode == 'batch':
    st.title("📈 骨架曲线提取器 - 批量处理模式")
    st.markdown("---")
    
    # 返回主页按钮
    if st.button("🏠 返回主页"):
        navigate_to('home')
    
    # 在侧边栏添加使用说明
    with st.sidebar:
        st.header("使用说明")
        st.markdown("""
        1. **选择文件夹**：输入待处理文件夹路径和输出文件夹路径
        2. **选择保存数据**：勾选需要保存的数据
        3. **开始批量处理**：点击按钮开始处理所有文件
        4. **等待完成**：处理完成后会有提示
        
        **注意事项：** 
        - 批量处理不会显示可视化结果
        - 支持.txt和.csv格式的文件
        - 输出文件会保存在指定的输出文件夹中
        """)
    
    st.header("📁 文件夹选择")
    
    # 选择输入和输出文件夹
    input_folder = st.text_input("输入文件夹路径（包含待处理的文件）:", "")
    output_folder = st.text_input("输出文件夹路径（处理结果保存位置）:", "")
    
    st.header("⚙️ 数据保存选项")
    
    # 处理选项
    col1, col2, col3 = st.columns(3)
    with col1:
        extract_peak_points = st.checkbox("骨架曲线峰值点数据", value=True)
    with col2:
        extract_envelope = st.checkbox("骨架曲线外包络线数据", value=True)
    with col3:
        smooth_processing = st.checkbox("骨架曲线平滑数据", value=True)
    
    # 如果选择了平滑处理，显示参数设置
    smooth_factor = 0.05  # 默认值
    num_points = 300      # 默认值
    
    if smooth_processing:
        colA1, colA2 = st.columns(2)
        
        with colA1:
            smooth_factor = st.slider(
                "RBF平滑因子",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="控制RBF插值的平滑程度"
            )
        
        with colA2:
            num_points = st.slider(
                "插值点数",
                min_value=100,
                max_value=500,
                value=300,
                step=50,
                help="平滑曲线上的点数"
            )
    
    # 开始批量处理按钮
    if st.button("🚀 开始批量处理", type="primary", width='stretch'):
        if not input_folder or not output_folder:
            st.warning("⚠️ 请输入完整的输入和输出文件夹路径")
            st.stop()
            
        if not os.path.exists(input_folder):
            st.error("❌ 输入文件夹不存在，请检查路径")
            st.stop()
            
        if not os.path.exists(output_folder):
            st.error("❌ 输出文件夹不存在，请检查路径")
            st.stop()
            
        if not (extract_peak_points or extract_envelope or smooth_processing):
            st.warning("⚠️ 请至少选择一种保存数据")
            st.stop()
            
        # 获取所有txt和csv文件
        txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
        all_files = txt_files + csv_files
        
        if not all_files:
            st.warning("⚠️ 输入文件夹中没有找到.txt或.csv文件")
            st.stop()
            
        try:
            with st.spinner(f"正在处理 {len(all_files)} 个文件，请稍候..."):
                # 使用封装好的批量处理函数
                success_count, failed_files = batch_process_files(
                    all_files, output_folder, 
                    extract_peak_points, extract_envelope, smooth_processing,
                    smooth_factor, num_points
                )
                        
                # 显示处理结果
                st.success(f"✅ 批量处理完成！成功处理 {success_count} 个文件")
                
                if failed_files:
                    st.warning(f"⚠️ {len(failed_files)} 个文件处理失败:")
                    for file_name, error in failed_files:
                        st.write(f"- {file_name}: {error}")
                        
        except Exception as e:
            st.error(f"❌ 批量处理过程中发生错误: {str(e)}")
            
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>骨架曲线提取器 Web版 - 批量处理模式</div>",
        unsafe_allow_html=True
    )
