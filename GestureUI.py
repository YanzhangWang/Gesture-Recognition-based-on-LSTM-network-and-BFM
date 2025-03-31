import tkinter as tk
import os
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import datetime

# 导入自定义模块
from ui_components import create_prediction_panel, create_visualization_panel, create_log_panel
from data_processor import DataProcessor
from gesture_model import GestureModel
from visualization import update_plot

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

class GestureRecognitionUI(tk.Tk):
    def __init__(self, background_image_path, model_path=None):
        super().__init__()

        # 设置窗口标题和大小
        self.title("WIFI-Based Gesture Recognition")
        self.geometry("1400x1000")
        self.configure(bg="#f0f0f0")

        # 添加背景图片
        self.background_image = Image.open(background_image_path)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(self, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        # 创建一个主框架来组织UI组件
        self.main_frame = tk.Frame(self, bg="white", bd=2, relief=tk.RIDGE)
        self.main_frame.place(relwidth=0.95, relheight=0.95, relx=0.025, rely=0.025)

        # 页面标题
        self.title_label = tk.Label(self.main_frame, text="WIFI-Based Gesture Recognition", 
                                   font=("Arial", 20, "bold"), bg="white", fg="#333333")
        self.title_label.pack(pady=10)

        # 创建上下两个部分的框架
        self.top_frame = tk.Frame(self.main_frame, bg="white")
        self.top_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.bottom_frame = tk.Frame(self.main_frame, bg="white")
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10, ipady =100)

        # 创建预测面板
        self.prediction_frame, self.prediction_elements = create_prediction_panel(self.top_frame)
        
        # 创建V矩阵可视化面板
        self.fig, self.ax = plt.subplots(2, 2, figsize=(6, 6))
        self.v_matrix_frame = create_visualization_panel(self.top_frame, self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.v_matrix_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建日志面板
        self.log_frame, self.log_text = create_log_panel(self.bottom_frame)
        
        # 初始化手势识别模型
        self.model = GestureModel(model_path) if model_path else None
        if self.model and self.model.is_loaded:
            self.log_message(f"Model loaded successfully: {os.path.basename(model_path)}")
            self.prediction_elements['status_label'].config(text="Model loaded - Ready")
        
        # 已处理的文件集合
        self.processed_files = set()
        
        # 启动数据处理器
        self.data_processor = None
        
    def log_message(self, message, error=False):
        """向日志窗口添加消息"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if error:
            self.log_text.insert(tk.END, f"[{timestamp}] ERROR: {message}\n", "error")
            self.log_text.tag_config("error", foreground="red")
        else:
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            
        self.log_text.see(tk.END)
        
    def update_prediction_display(self, result):
        """更新预测结果显示"""
        if not result:
            return
        
        # 在适当的位置添加静态检测信息显示
        # 更新状态

        elements = self.prediction_elements

        if 'is_static_detected' in result:
            is_static = result['is_static_detected']
            static_status = "静态" if is_static else "动态"
            static_index = result.get('static_index', 'N/A')
            static_threshold = result.get('static_threshold', 'N/A')
            elements['status_label'].config(
                text=f"处理: {os.path.basename(result['file'])} ({static_status}, 指数: {static_index:.6f}, 阈值: {static_threshold})",
                fg="#006600"
            )
        else:
            elements['status_label'].config(
                text=f"处理: {os.path.basename(result['file'])}",
                fg="#006600"
            )

        # 更新当前识别的手势
        elements['current_gesture'].config(text=result["predicted_label"].capitalize(), fg="#0066cc")
        
        # 更新置信度
        elements['confidence_value'].config(text=f"{result['confidence']:.2f}%")
        
        # 更新概率分布进度条
        for label, (progress_bar, value_label) in elements['progress_bars'].items():
            probability = result["probabilities"].get(label, 0)
            progress_bar["value"] = probability
            value_label.config(text=f"{probability:.2f}%")
            
        
        # 添加到历史记录
        self.add_history_item(
            result["predicted_label"].capitalize(),
            os.path.basename(result['file']),
            result["confidence"]
        )
        
    def add_history_item(self, gesture, filename, confidence):
        """向历史记录添加一个预测项目"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        elements = self.prediction_elements
        
        # 创建一个项目框架
        item_frame = tk.Frame(elements['history_scroll_frame'], bg="#f0f8ff", bd=1, relief=tk.GROOVE)
        item_frame.pack(fill=tk.X, pady=2, padx=2)
        
        # 时间戳
        tk.Label(item_frame, text=timestamp, bg="#f0f8ff", width=8).pack(side=tk.LEFT, padx=2)
        
        # 文件名
        tk.Label(item_frame, text=filename, bg="#f0f8ff", width=15).pack(side=tk.LEFT, padx=2)
        
        # 手势
        gesture_label = tk.Label(item_frame, text=gesture, bg="#f0f8ff", width=10)
        gesture_label.pack(side=tk.LEFT, padx=2)
        
        # 根据不同手势类型设置不同的颜色
        if gesture == "Updown":
            gesture_label.config(fg="#0066cc")
        elif gesture == "Leftright":
            gesture_label.config(fg="#cc6600")
        elif gesture == "Empty":
            gesture_label.config(fg="#666666")
            
        # 置信度
        tk.Label(item_frame, text=f"{confidence:.2f}%", bg="#f0f8ff", width=8).pack(side=tk.LEFT, padx=2)
        
        # 滚动到最新项目
        elements['history_canvas'].update_idletasks()
        elements['history_canvas'].yview_moveto(1.0)
    
    def start_processing(self, args):
        """启动数据处理"""
        self.data_processor = DataProcessor(
            self, 
            self.model,
            args,
            self.log_message,
            self.update_prediction_display,
            self.ax,
            self.canvas
        )
        self.data_processor.start()
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="WIFI-Based Gesture Recognition")
    parser.add_argument('standard', help='which standard are you operating on, options are "AC" or "AX"')
    parser.add_argument('mimo', help='which type of network are you forming, options are "SU" for su-mimo or "MU" for mu-mimo')
    parser.add_argument('config', help='which type of antenna config you have, for now, available options are 3x1 with AC and 4x2 with AX')
    parser.add_argument('bw', help='bandwidth of the capture')
    parser.add_argument('MAC', help='MAC of the Target Device')
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('seconds_interval', help='Seconds every which you have a new file', type=int)
    parser.add_argument('--model', help='Path to model file', default=None)
    parser.add_argument('--background', help='Path to background image', default=None)
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置默认值
    MODEL_PATH = args.model or '/home/ggbo/FYP/Python_code/network_models/finger__TX[0, 1, 2, 3]_RX[0, 1]_posTRAIN[1, 2, 3, 4, 5, 6, 7, 8, 9]_posTEST[1, 2, 3, 4, 5, 6, 7, 8, 9]_bandwidth80convolutional_S1_20epochnetwork.h5'
    BACKGROUND_PATH = args.background or "/home/ggbo/Downloads/e34b40f1d8a3492b86e6876eeee41432.webp"
    
    # 启动应用
    app = GestureRecognitionUI(BACKGROUND_PATH, MODEL_PATH)
    app.log_message(f"Starting application with settings: {args.standard} {args.mimo} {args.config} {args.bw} {args.dir}")
    
    # 启动数据处理
    app.start_processing(args)
    
    # 运行主循环
    app.mainloop()
