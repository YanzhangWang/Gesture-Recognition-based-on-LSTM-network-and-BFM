import tkinter as tk
from tkinter import scrolledtext, ttk, Frame, Label

# 手势标签映射
label_mapping = {
    0: 'empty',
    1: 'leftright', 
    2: 'updown',
}

def create_prediction_panel(parent):
    """创建预测结果显示面板"""
    prediction_frame = tk.Frame(parent, bg="white", bd=2, relief=tk.GROOVE)
    prediction_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

    # 预测结果标题
    predict_title = tk.Label(prediction_frame, text="Gesture Recognition Results", 
                           font=("Arial", 14, "bold"), bg="white", fg="#333333")
    predict_title.pack(pady=5)

    # 创建结果显示区
    results_frame = tk.Frame(prediction_frame, bg="white")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # 当前预测
    current_gesture_frame = tk.Frame(results_frame, bg="white")
    current_gesture_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(current_gesture_frame, text="Current Gesture:", 
            font=("Arial", 12), bg="white").pack(side=tk.LEFT, padx=5)
    
    current_gesture = tk.Label(current_gesture_frame, text="Waiting...", 
                              font=("Arial", 14, "bold"), bg="white", fg="#0066cc")
    current_gesture.pack(side=tk.LEFT, padx=5)

    # 置信度
    confidence_frame = tk.Frame(results_frame, bg="white")
    confidence_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(confidence_frame, text="Confidence:", 
            font=("Arial", 12), bg="white").pack(side=tk.LEFT, padx=5)
    
    confidence_value = tk.Label(confidence_frame, text="0.00%", 
                                font=("Arial", 12), bg="white", fg="#0066cc")
    confidence_value.pack(side=tk.LEFT, padx=5)

    # 创建进度条显示各类别概率
    probabilities_frame = tk.Frame(results_frame, bg="white")
    probabilities_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    tk.Label(probabilities_frame, text="Probability Distribution:", 
            font=("Arial", 12), bg="white").pack(anchor=tk.W, padx=5, pady=5)
    
    # 为每个手势类别创建进度条
    progress_bars = {}
    for i, label in label_mapping.items():
        frame = tk.Frame(probabilities_frame, bg="white")
        frame.pack(fill=tk.X, pady=2)
        
        tk.Label(frame, text=f"{label.capitalize()}:", 
                font=("Arial", 10), bg="white", width=10, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        progress = ttk.Progressbar(frame, length=300, mode='determinate')
        progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        value_label = tk.Label(frame, text="0.00%", font=("Arial", 10), bg="white", width=8)
        value_label.pack(side=tk.LEFT, padx=5)
        
        progress_bars[label] = (progress, value_label)

    # 文件处理状态
    status_frame = tk.Frame(results_frame, bg="white")
    status_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(status_frame, text="Status:", 
            font=("Arial", 12), bg="white").pack(side=tk.LEFT, padx=5)
    
    status_label = tk.Label(status_frame, text="Waiting for data...", 
                          font=("Arial", 12), bg="white", fg="#666666")
    status_label.pack(side=tk.LEFT, padx=5)

    # 处理历史记录
    history_frame = tk.Frame(results_frame, bg="white", bd=1, relief=tk.GROOVE)
    history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    tk.Label(history_frame, text="Prediction History", 
            font=("Arial", 11, "bold"), bg="white").pack(anchor=tk.W, padx=5, pady=5)
    
    # 创建历史记录列表框架
    history_list_frame = tk.Frame(history_frame, bg="white")
    history_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 添加滚动区域用于显示历史记录
    history_canvas = tk.Canvas(history_list_frame, bg="white", highlightthickness=0)
    scrollbar = ttk.Scrollbar(history_list_frame, orient="vertical", command=history_canvas.yview)
    history_scroll_frame = tk.Frame(history_canvas, bg="white")
    
    history_scroll_frame.bind(
        "<Configure>",
        lambda e: history_canvas.configure(scrollregion=history_canvas.bbox("all"))
    )
    
    history_canvas.create_window((0, 0), window=history_scroll_frame, anchor="nw")
    history_canvas.configure(yscrollcommand=scrollbar.set)
    
    history_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # 返回预测面板和各元素引用
    prediction_elements = {
        'current_gesture': current_gesture,
        'confidence_value': confidence_value,
        'status_label': status_label,
        'progress_bars': progress_bars,
        'history_canvas': history_canvas,
        'history_scroll_frame': history_scroll_frame
    }
    
    return prediction_frame, prediction_elements

def create_visualization_panel(parent, figure=None):
    """创建V矩阵可视化面板"""
    v_matrix_frame = tk.Frame(parent, bg="white", bd=2, relief=tk.GROOVE)
    v_matrix_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
    
    # V矩阵标题
    v_matrix_title = tk.Label(v_matrix_frame, text="V-Matrix Visualization", 
                             font=("Arial", 14, "bold"), bg="white", fg="#333333")
    v_matrix_title.pack(pady=5)
    
    return v_matrix_frame

def create_log_panel(parent):
    """创建日志面板"""
    log_frame = tk.Frame(parent, bg="white", bd=2, relief=tk.GROOVE)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 日志标题
    log_title = tk.Label(log_frame, text="System Log", 
                        font=("Arial", 14, "bold"), bg="white", fg="#333333")
    log_title.pack(pady=5)
    
    # 创建外部框架用于包含日志文本和滚动条
    log_container = tk.Frame(log_frame, bg="white")
    log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # 创建滚动文本区域来显示日志
    log_text = scrolledtext.ScrolledText(
        log_container, 
        wrap=tk.WORD, 
        bg="#f9f9f9", 
        fg="#333333", 
        font=("Consolas", 11),  # 增大字体
        height=20,              # 增大高度
        padx=8,                 # 增加内边距
        pady=8
    )
    log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
    
    # 添加自定义样式的滚动条
    style = ttk.Style()
    style.configure("Bigger.Vertical.TScrollbar", 
                    gripcount=0, 
                    background="#aaaaaa", 
                    troughcolor="#f0f0f0",
                    arrowsize=20,          # 增大箭头大小
                    width=20)              # 增大滚动条宽度
    
    # 创建更大的滚动条并连接到文本框
    scrollbar = ttk.Scrollbar(
        log_container, 
        style="Bigger.Vertical.TScrollbar",
        orient="vertical", 
        command=log_text.yview
    )
    scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
    log_text.config(yscrollcommand=scrollbar.set)
    
    return log_frame, log_text
