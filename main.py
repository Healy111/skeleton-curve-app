from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from 骨架曲线提取器 import *


class SkeletonCurveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("骨架曲线提取器")
        self.root.geometry("500x600")
        self.root.resizable(False, False)

        # 设置界面
        self.setup_ui()

        # 初始化变量
        self.file_path = None

    def setup_ui(self):
        default_font = ("Microsoft YaHei", 10)
        title_font = ("Microsoft YaHei", 18, "bold")
        button_font = ("Microsoft YaHei", 12, "bold")

        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=20, fill=tk.X, padx=30)

        title_label = ttk.Label(
            title_frame,
            text="骨架曲线提取器",
            font=title_font,
            foreground="#2C3E50"
        )
        title_label.pack(anchor=tk.CENTER)

        # 分隔线
        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=30, pady=5)

        # 文件选择框架
        file_frame = ttk.LabelFrame(self.root, text="文件选择", padding=15)
        file_frame.pack(pady=15, fill=tk.X, padx=30)

        ttk.Label(
            file_frame,
            text="请选择滞回曲线数据文件：",
            font=default_font
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        # 文件路径输入框
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(
            file_frame,
            textvariable=self.file_path_var,
            width=50,
            state="readonly",
            font=default_font
        )
        self.file_path_entry.grid(row=1, column=0, sticky=tk.EW, padx=(0, 10))

        # 浏览按钮
        browse_button = ttk.Button(
            file_frame,
            text="浏览...",
            command=self.browse_file,
            width=10
        )
        browse_button.grid(row=1, column=1, sticky=tk.W)

        file_frame.columnconfigure(0, weight=1)

        # 处理按钮
        self.process_button = ttk.Button(
            self.root,
            text="提取骨架曲线",
            command=self.process_data,
            style="Accent.TButton"
        )
        self.process_button.pack(pady=20)

        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("请选择一个滞回曲线数据文件")
        self.status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 10),
            foreground="blue"
        )
        self.status_label.pack(pady=10)

        # 信息文本框
        info_frame = ttk.LabelFrame(self.root, text="使用说明", padding=10)
        info_frame.pack(pady=10, fill=tk.X, padx=30)

        info_text = tk.Text(info_frame, height=4, width=60, wrap=tk.WORD, font=("Microsoft YaHei", 9))
        info_text.pack()
        info_text.insert(tk.END, "1. 点击'浏览...'按钮选择滞回曲线数据文件\n"
                                 "2. 数据文件应为.txt格式，包含位移和力两列数据\n"
                                 "3. 点击'提取骨架曲线'按钮开始处理\n"
                                 "4. 处理完成后将显示结果图表")
        info_text.config(state=tk.DISABLED)

    def browse_file(self):
        """浏览并选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择滞回曲线数据文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.status_var.set("文件已选择，点击'提取骨架曲线'开始处理")

    def process_data(self):
        """处理数据"""
        if not self.file_path:
            messagebox.showwarning("警告", "请先选择一个数据文件")
            return

        try:
            self.status_var.set("正在处理数据...")
            self.root.update()

            # 调用骨架曲线提取器处理数据
            main_process_with_file(self.file_path)

            self.status_var.set("处理完成！")

        except Exception as e:
            messagebox.showerror("错误", f"处理过程中发生错误:\n{str(e)}")
            self.status_var.set("处理失败")


def main_process_with_file(file_path):
    """使用指定文件路径处理数据的函数"""
    # 读取txt文件
    displacement, force = read_txt_data(file_path)

    # 创建骨架曲线提取器
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


if __name__ == "__main__":
    root = tk.Tk()
    app = SkeletonCurveGUI(root)
    root.mainloop()
