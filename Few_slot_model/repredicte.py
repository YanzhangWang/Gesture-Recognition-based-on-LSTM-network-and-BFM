import tensorflow as tf
import numpy as np
import argparse
import os

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
        
        print(f"使用特征维度: {self.flat_dim}")
        
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
                print(f"已复制特征以匹配期望的维度: {x_flat.shape}")
            
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

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    try:
        # 加载.npy文件
        data = np.load(file_path, allow_pickle=True)
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
        
        return processed
        
    except Exception as e:
        print(f"\n数据处理失败: {str(e)}")
        print("常见解决方法：")
        print("1. 确保文件为.npy格式")
        print("2. 确保数据形状符合(234, 4, 2)要求")
        print("3. 检查数据采集是否正确")
        exit(1)

def predict_with_prototype(model_weights_path, file_path, support_data_path, input_shape=(234, 4, 2), num_classes=3):
    """
    使用原型网络模型进行预测
    
    参数:
        model_weights_path: 模型权重文件路径
        file_path: 待预测数据文件路径
        support_data_path: 支持集数据目录
        input_shape: 输入数据形状
        num_classes: 类别数量
    """
    # 创建原型网络模型
    print("创建原型网络模型...")
    proto_model = PrototypicalNetwork(input_shape, use_double_dim=True)  # 使用双倍维度
    
    # 构建模型（需要一次前向传播来初始化权重）
    # 打印输入形状以调试
    print(f"输入形状: {input_shape}, 计算的扁平维度: {proto_model.flat_dim}")
    
    # 创建虚拟输入进行初始化
    dummy_support = np.zeros((3, *input_shape))
    dummy_support_labels = np.array([0, 1, 2])
    dummy_query = np.zeros((1, *input_shape))
    
    # 验证特征提取过程
    support_features = proto_model.extract_features(dummy_support)
    print(f"提取特征后的形状: {support_features.shape}")
    
    # 测试模型初始化
    try:
        proto_model(dummy_support, dummy_support_labels, dummy_query, num_classes, 1)
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)
    
    # 加载模型权重
    try:
        proto_model.load_weights(model_weights_path)
        print(f"模型权重加载成功: {model_weights_path}")
    except Exception as e:
        print(f"模型权重加载失败: {str(e)}")
        exit(1)
    
    # 构建支持集
    print("\n构建支持集...")
    support_set_x = []
    support_set_y = []
    
    # 每个类别选择一个样本作为支持集
    k_shot = 1  # 每类只用1个样本
    
    # 简化：直接使用预定义的示例文件
    support_files = {
        0: os.path.join(support_data_path, "vmatrices_empty_1.npy"),
        1: os.path.join(support_data_path, "vmatrices_stop_1.npy"),
        2: os.path.join(support_data_path, "vmatrices_first_1.npy")
    }
    
    for class_idx, support_file in support_files.items():
        try:
            if os.path.exists(support_file):
                support_data = load_and_preprocess_data(support_file)
                support_set_x.append(support_data[0])  # 去掉批次维度
                support_set_y.append(class_idx)
                print(f"已加载类别 {label_mapping[class_idx]} 的支持样本: {os.path.basename(support_file)}")
            else:
                print(f"警告: 找不到类别 {label_mapping[class_idx]} 的支持样本文件: {support_file}")
        except Exception as e:
            print(f"加载支持样本 {support_file} 失败: {e}")
    
    if not support_set_x:
        print("错误: 无法加载任何支持样本。请检查支持集数据路径。")
        exit(1)
    
    support_set_x = np.array(support_set_x)
    support_set_y = np.array(support_set_y)
    print(f"支持集形状: {support_set_x.shape}, 标签形状: {support_set_y.shape}")
    
    # 加载待预测数据
    query_data = load_and_preprocess_data(file_path)
    print(f"\n预测样本形状: {query_data.shape}")
    
    # 执行预测
    try:
        logits = proto_model(support_set_x, support_set_y, query_data, num_classes, k_shot, training=False)
        predictions = tf.nn.softmax(logits, axis=1).numpy()[0]
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 解析结果
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100  # 转换为百分比
    
    # 输出结果
    print("\n" + "="*40)
    print(f"文件: {os.path.basename(file_path)}")
    print(f"预测类型: {label_mapping.get(predicted_class, f'未知({predicted_class})')} (编号: {predicted_class})")
    print(f"置信度: {confidence:.2f}%")
    print("各类概率分布:")
    for i, prob in enumerate(predictions):
        if i < len(label_mapping):
            label = label_mapping.get(i, f"类别{i}")
            print(f"  {label}: {prob * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description='原型网络预测 - WiFi波束成形矩阵分类')
    parser.add_argument('model_path', type=str, help='模型权重文件路径 (.h5)')
    parser.add_argument('data_path', type=str, help='要预测的数据文件路径 (.npy)')
    parser.add_argument('--support_dir', type=str, default='/home/ggbo/static_dataset/Vmatrices/', 
                      help='支持集数据目录（包含各类别样本）')
    parser.add_argument('--num_classes', type=int, default=3, help='类别数量')
    args = parser.parse_args()
    
    # 执行预测
    predict_with_prototype(
        args.model_path, 
        args.data_path,
        args.support_dir,
        input_shape=(234, 4, 2),
        num_classes=args.num_classes
    )