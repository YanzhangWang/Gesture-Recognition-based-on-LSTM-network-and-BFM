import tensorflow as tf
import numpy as np
import argparse
import os
import time
import glob
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 添加少样本学习模型定义
class PrototypicalNetwork(tf.keras.Model):
    """
    原型网络实现，用于WiFi波束成形矩阵的少样本学习
    """
    def __init__(self, input_shape, use_double_dim=True):
        super(PrototypicalNetwork, self).__init__()
        
        # 计算扁平后的特征大小
        if use_double_dim:
            self.flat_dim = np.prod(input_shape) * 2  # 与原始训练模型保持一致
        else:
            self.flat_dim = np.prod(input_shape)
        
        # 创建特征提取器 - 定义明确的输入维度
        self.encoder = tf.keras.Sequential([
            # 第一个层明确指定输入维度
            tf.keras.layers.Dense(256, activation=None, input_shape=(self.flat_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2)
        ])
    
    def extract_features(self, x):
        """预处理复数数据并提取特征"""
        
        # 转换复数数据为实数特征
        if hasattr(x, 'dtype') and 'complex' in str(x.dtype).lower():
            # 提取实部和虚部
            real_part = tf.math.real(x)
            imag_part = tf.math.imag(x)
            
            # 展平并拼接
            real_flat = tf.reshape(real_part, [tf.shape(x)[0], -1])
            imag_flat = tf.reshape(imag_part, [tf.shape(x)[0], -1])
            features = tf.concat([real_flat, imag_flat], axis=-1)
            
            return tf.cast(features, tf.float32)
        
        elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.complex_):
            # 处理NumPy复数数组
            real_part = np.real(x)
            imag_part = np.imag(x)
            
            # 展平并拼接
            real_flat = np.reshape(real_part, (x.shape[0], -1))
            imag_flat = np.reshape(imag_part, (x.shape[0], -1))
            features = np.concatenate([real_flat, imag_flat], axis=-1)
            
            return tf.convert_to_tensor(features, dtype=tf.float32)
        
        else:
            # 处理已经是实数的数据
            x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
            
            # 如果使用双倍维度，通过复制数据模拟原始模型的输入格式
            if hasattr(self, 'flat_dim') and self.flat_dim == np.prod(x.shape[1:]) * 2:
                x_flat = tf.concat([x_flat, x_flat], axis=-1)
                
            return x_flat
    
    def call(self, support_set, support_labels, query_set, n_way, k_shot, training=False):
        """
        前向传播
        
        参数:
            support_set: 支持集样本，形状为[n_support, ...]
            support_labels: 支持集标签，形状为[n_support]
            query_set: 查询集样本，形状为[n_query, ...]
            n_way: 分类任务的类别数
            k_shot: 每个类别的样本数
            training: 是否处于训练模式
        """
        # 处理支持集
        support_features = self.extract_features(support_set)
        z_support = self.encoder(support_features, training=training)
        
        # 处理查询集
        query_features = self.extract_features(query_set)
        z_query = self.encoder(query_features, training=training)
        
        # 计算原型
        prototypes = []
        support_labels_np = support_labels
        
        if isinstance(support_labels, tf.Tensor):
            support_labels_np = support_labels.numpy()
        
        for i in range(n_way):
            # 找到属于当前类别的样本
            class_indices = np.where(support_labels_np == i)[0]
            class_samples = tf.gather(z_support, class_indices)
            
            # 计算类别原型（均值）
            prototype = tf.reduce_mean(class_samples, axis=0)
            prototypes.append(prototype)
        
        # 将原型堆叠成一个张量
        prototypes = tf.stack(prototypes)  # [n_way, feature_dim]
        
        # 计算查询样本到每个原型的欧几里得距离
        dists = []
        for prototype in prototypes:
            # 计算到这个原型的距离
            dist = tf.reduce_sum(tf.square(z_query - tf.expand_dims(prototype, 0)), axis=1)
            dists.append(dist)
        
        dists = tf.stack(dists, axis=1)  # [n_query, n_way]
        
        # 返回负距离作为logits
        return -dists

# 标签映射
label_mapping = {
    0: 'empty',
    1: 'stop',
    2: 'first'
}

def softmax(x):
    """ 计算 softmax，确保输出概率正常 """
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x)

def load_and_preprocess_data(file_path, verbose=True):
    """
    加载并预处理数据
    """
    try:
        # 加载.npy文件
        data = np.load(file_path, allow_pickle=True)
        if verbose:
            print(f"\n处理文件: {os.path.basename(file_path)}")
            print(f"文件大小: {os.path.getsize(file_path)/1024:.2f} KB")
            print(f"数据形状: {data.shape}")
        
        # 数据形状检查与处理
        if len(data.shape) == 3 and data.shape == (234, 4, 2):
            # 单样本，添加批次维度
            processed = np.expand_dims(data, axis=0)
        elif len(data.shape) == 4 and data.shape[1:] == (234, 4, 2):
            # 多样本，只取第一个
            processed = data[0:1]
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
        
        # 数据标准化
        mean = np.mean(processed)
        std = np.std(processed)
        std = std if std > 0 else 1e-6  # 防止除零
        processed = (processed - mean) / std
        
        if verbose:
            print(f"数据范围: [{np.min(processed):.2f}, {np.max(processed):.2f}]")
            print(f"数据均值: {np.mean(processed):.2f} ± {np.std(processed):.2f}")
        
        # 检查并处理NaN和Inf
        if np.isnan(processed).any() or np.isinf(processed).any():
            if verbose:
                print("发现NaN或Inf值，已替换")
            processed = np.nan_to_num(processed, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return processed
        
    except Exception as e:
        print(f"\n数据处理失败: {str(e)}")
        print("常见解决方法：")
        print("1. 确保文件为.npy格式")
        print("2. 确保数据形状符合(234, 4, 2)要求")
        print("3. 检查数据采集是否正确")
        return None

def predict_with_prototype(proto_model, file_path, support_set_x, support_set_y, num_classes=3, verbose=True):
    """
    使用原型网络模型进行预测
    
    参数:
        proto_model: 原型网络模型
        file_path: 待预测数据文件路径
        support_set_x: 支持集样本
        support_set_y: 支持集标签
        num_classes: 类别数量
        verbose: 是否显示详细输出
    """
    # 加载待预测数据
    query_data = load_and_preprocess_data(file_path, verbose=verbose)
    
    if query_data is None:
        return None
        
    if verbose:
        print(f"\n预测样本形状: {query_data.shape}")
    
    # 执行预测
    try:
        logits = proto_model(support_set_x, support_set_y, query_data, num_classes, 1, training=False)
        predictions = tf.nn.softmax(logits, axis=1).numpy()[0]
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 解析结果
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100  # 转换为百分比
    
    # 输出结果
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*40)
    print(f"时间: {timestamp}")
    print(f"文件: {os.path.basename(file_path)}")
    print(f"预测类型: {label_mapping.get(predicted_class, f'未知({predicted_class})')} (编号: {predicted_class})")
    print(f"置信度: {confidence:.2f}%")
    print("各类概率分布:")
    for i, prob in enumerate(predictions):
        if i < len(label_mapping):
            label = label_mapping.get(i, f"类别{i}")
            print(f"  {label}: {prob * 100:.2f}%")
    print("="*40 + "\n")
    
    # 返回预测结果
    return {
        "timestamp": timestamp,
        "file": os.path.basename(file_path),
        "predicted_class": predicted_class,
        "predicted_label": label_mapping.get(predicted_class, f'未知({predicted_class})'),
        "confidence": confidence,
        "probabilities": {label_mapping.get(i, f"类别{i}"): prob * 100 for i, prob in enumerate(predictions)}
    }

def build_support_set(support_data_path, verbose=True):
    """
    构建支持集
    
    参数:
        support_data_path: 支持集数据目录
        verbose: 是否显示详细输出
    """
    if verbose:
        print("\n构建支持集...")
    
    support_set_x = []
    support_set_y = []
    
    # 每个类别选择一个样本作为支持集
    # 支持集文件
    support_files = {
        0: os.path.join(support_data_path, "vmatrices_empty_1.npy"),
        1: os.path.join(support_data_path, "vmatrices_stop_1.npy"),
        2: os.path.join(support_data_path, "vmatrices_first_1.npy")
    }
    
    for class_idx, support_file in support_files.items():
        try:
            if os.path.exists(support_file):
                support_data = load_and_preprocess_data(support_file, verbose=verbose)
                support_set_x.append(support_data[0])  # 去掉批次维度
                support_set_y.append(class_idx)
                if verbose:
                    print(f"已加载类别 {label_mapping[class_idx]} 的支持样本: {os.path.basename(support_file)}")
            else:
                print(f"警告: 找不到类别 {label_mapping[class_idx]} 的支持样本文件: {support_file}")
        except Exception as e:
            print(f"加载支持样本 {support_file} 失败: {e}")
    
    if not support_set_x:
        print("错误: 无法加载任何支持样本。请检查支持集数据路径。")
        return None, None
    
    support_set_x = np.array(support_set_x)
    support_set_y = np.array(support_set_y)
    
    if verbose:
        print(f"支持集形状: {support_set_x.shape}, 标签形状: {support_set_y.shape}")
    
    return support_set_x, support_set_y

class FileHandler(FileSystemEventHandler):
    def __init__(self, proto_model, support_set_x, support_set_y, watch_dir, num_classes=3, file_pattern='*.npy', verbose=True):
        self.proto_model = proto_model
        self.support_set_x = support_set_x
        self.support_set_y = support_set_y
        self.watch_dir = watch_dir
        self.num_classes = num_classes
        self.file_pattern = file_pattern
        self.verbose = verbose
        self.processed_files = set()
        # 处理已有文件
        self._process_existing_files()
        
    def _process_existing_files(self):
        """处理监控目录中已有的文件"""
        existing_files = glob.glob(os.path.join(self.watch_dir, self.file_pattern))
        for file_path in existing_files:
            self.processed_files.add(os.path.basename(file_path))
            
    def on_created(self, event):
        """当新文件创建时调用"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not self._is_target_file(file_path):
            return
            
        # 避免重复处理
        file_name = os.path.basename(file_path)
        if file_name in self.processed_files:
            return
            
        # 等待文件写入完成
        self._wait_for_file_ready(file_path)
        
        # 处理文件
        print(f"\n发现新文件: {file_name}")
        predict_with_prototype(
            self.proto_model, 
            file_path, 
            self.support_set_x, 
            self.support_set_y, 
            self.num_classes,
            verbose=False
        )
        self.processed_files.add(file_name)
        
    def _is_target_file(self, file_path):
        """检查文件是否匹配目标格式"""
        return file_path.endswith('.npy') 
        
    def _wait_for_file_ready(self, file_path, timeout=5, check_interval=0.1):
        """等待文件写入完成"""
        start_time = time.time()
        last_size = -1
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(file_path)
                if current_size > 0 and current_size == last_size:
                    # 文件大小稳定，认为写入完成
                    return True
                last_size = current_size
                time.sleep(check_interval)
            except OSError:
                # 文件可能暂时无法访问
                time.sleep(check_interval)
                
        # 超时但仍然返回，尝试处理文件
        return True

def real_time_monitor(proto_model, support_set_x, support_set_y, watch_dir, num_classes=3, file_pattern='*.npy', verbose=True):
    """实时监控目录中的新文件进行预测"""
    print(f"开始监控目录: {watch_dir}")
    print(f"监控文件类型: {file_pattern}")
    print("等待新数据文件...")
    
    # 创建观察者和处理器
    event_handler = FileHandler(proto_model, support_set_x, support_set_y, watch_dir, num_classes, file_pattern, verbose)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n监控已停止")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # 直接在代码中设置参数
    # =================================================================
    # 必需参数
    MODEL_PATH = '/home/ggbo/FYP/Python_code/network_models/few_slot_model_TX[0, 1, 2, 3]_RX[0, 1]_posTRAIN[1, 2, 3, 4, 5, 6, 7, 8, 9]_posTEST[1, 2, 3, 4, 5, 6, 7, 8, 9]_bandwidth80_prototype_prototype.h5'  # 修改为你的模型路径
    INPUT_SHAPE = (234, 4, 2)
    NUM_CLASSES = 3
    SUPPORT_DIR = '/home/ggbo/static_dataset/Vmatrices/'  # 支持集数据目录
    
    # 可选参数
    WATCH_DIRECTORY = '/home/ggbo/Wi-BFI/Demo/BFM_data/vmatrix'  # 监控目录
    FILE_PATTERN = '*.npy'       # 文件匹配模式
    SINGLE_FILE = None           # 单文件预测模式, 设为None启用监控模式
    VERBOSE = True               # 是否显示详细输出
    # =================================================================

    # 如果你仍需保留命令行参数选项
    parser = argparse.ArgumentParser(description='原型网络实时预测')
    parser.add_argument('--model_path', type=str, help='模型权重文件路径 (.h5)', default=MODEL_PATH)
    parser.add_argument('--support_dir', type=str, help='支持集数据目录', default=SUPPORT_DIR)
    parser.add_argument('--watch_dir', type=str, default=WATCH_DIRECTORY, help='监控目录路径')
    parser.add_argument('--pattern', type=str, default=FILE_PATTERN, help='文件匹配模式')
    parser.add_argument('--single_file', type=str, default=SINGLE_FILE, help='单文件预测模式')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='类别数量')
    parser.add_argument('--verbose', action='store_true', default=VERBOSE, help='显示详细输出')
    
    # 解析命令行参数，命令行参数会覆盖代码中设置的默认值
    args = parser.parse_args()
    
    # 使用设置好的参数
    model_path = args.model_path
    support_dir = args.support_dir
    watch_dir = args.watch_dir
    pattern = args.pattern
    single_file = args.single_file
    num_classes = args.num_classes
    verbose = args.verbose

    # 创建原型网络模型
    print("创建原型网络模型...")
    proto_model = PrototypicalNetwork(INPUT_SHAPE, use_double_dim=True)
    
    # 构建模型（需要一次前向传播来初始化权重）
    if verbose:
        print(f"输入形状: {INPUT_SHAPE}, 计算的扁平维度: {proto_model.flat_dim}")
    
    # 创建虚拟输入进行初始化
    dummy_support = np.zeros((3, *INPUT_SHAPE))
    dummy_support_labels = np.array([0, 1, 2])
    dummy_query = np.zeros((1, *INPUT_SHAPE))
    
    # 初始化模型
    try:
        proto_model(dummy_support, dummy_support_labels, dummy_query, num_classes, 1)
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)
    
    # 加载模型权重
    try:
        proto_model.load_weights(model_path)
        print(f"模型权重加载成功: {model_path}")
    except Exception as e:
        print(f"模型权重加载失败: {str(e)}")
        exit(1)
    
    # 构建支持集
    support_set_x, support_set_y = build_support_set(support_dir, verbose=verbose)
    
    if support_set_x is None or support_set_y is None:
        print("无法构建支持集，退出")
        exit(1)

    # 执行模式
    if single_file:
        # 单文件预测模式
        predict_with_prototype(proto_model, single_file, support_set_x, support_set_y, num_classes, verbose=verbose)
    else:
        # 实时监控模式
        real_time_monitor(proto_model, support_set_x, support_set_y, watch_dir, num_classes, pattern, verbose)