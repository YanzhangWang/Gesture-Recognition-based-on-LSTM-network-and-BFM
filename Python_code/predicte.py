import tensorflow as tf
import numpy as np
import argparse
import os

# 自定义3D卷积层
class ConvNormalization3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='same', activation='relu',
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', bn=True, name_layer=None, **kwargs):
        super(ConvNormalization3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name_layer = name_layer
        self.conv_l = tf.keras.layers.Conv3D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding,
                                             name=self.name_layer, kernel_initializer=self.kernel_initializer,
                                             bias_initializer=self.bias_initializer)
        self.bn = bn
        self.activation = activation
        if bn:
            bn_name = None if self.name_layer is None else self.name_layer + '_bn'
            self.bn_l = tf.keras.layers.BatchNormalization(axis=4, name=bn_name)
        if activation is not None:
            self.act_l = tf.keras.layers.Activation(self.activation)

    def call(self, x_in):
        x = self.conv_l(x_in)
        if self.bn:
            x = self.bn_l(x)
        if self.activation is not None:
            x = self.act_l(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
             'filters': self.filters,
             'kernel_size': self.kernel_size,
             'strides': self.strides,
             'padding': self.padding,
             'activation': self.activation,
             'kernel_initializer': self.kernel_initializer,
             'bias_initializer': self.bias_initializer,
             'bn': self.bn,
             'name_layer': self.name_layer
        })
        return config

# 原始的ConvNormalization类（用于兼容旧模型）
class ConvNormalization(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation='selu',
                 kernel_initializer='lecun_normal', bias_initializer='zeros', bn=True, name_layer=None, **kwargs):
        super(ConvNormalization, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name_layer = name_layer
        self.conv_l = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding,
                                             name=self.name_layer, kernel_initializer=self.kernel_initializer,
                                             bias_initializer=self.bias_initializer)
        self.bn = bn
        self.activation = activation
        if bn:
            bn_name = None if self.name_layer is None else self.name_layer + '_bn'
            self.bn_l = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)
        if activation is not None:
            self.act_l = tf.keras.layers.Activation(self.activation)

    def call(self, x_in):
        x = self.conv_l(x_in)
        if self.bn:
            x = self.bn_l(x)
        if self.activation is not None:
            x = self.act_l(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
             'filters': self.filters,
             'kernel_size': self.kernel_size,
             'strides': self.strides,
             'padding': self.padding,
             'activation': self.activation,
             'kernel_initializer': self.kernel_initializer,
             'bias_initializer': self.bias_initializer,
             'bn': self.bn,
             'name_layer': self.name_layer
        })
        return config

# 标签映射
label_mapping = {
    0: 'empty',
    1: 'leftright',
    2: 'updown'
    # 0: 'updown',
    # 1: 'leftright', 
    # 2: 'stand',
    # 3: 'empty'
}

def softmax(x):
    """ 计算 softmax，确保输出概率正常 """
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x)

def load_and_preprocess_3d(file_path):
    """
    加载并预处理数据用于3D CNN+LSTM模型
    """
    try:
        # 检查文件格式
        if file_path.endswith('.npy'):
            # 直接加载.npy文件
            try:
                data = np.load(file_path, allow_pickle=True)
                print(f"\n处理文件: {os.path.basename(file_path)}")
                print(f"文件大小: {os.path.getsize(file_path)/1024:.2f} KB")
                print(f"数据形状: {data.shape}")
                
                # 如果是(500, 234, 4, 2)的格式，取第一个样本
                if len(data.shape) == 4 and data.shape[1:] == (234, 4, 2):
                    processed = data[0:1]  # 取第一个样本
                else:
                    print(f"警告: 数据形状不是预期的(N, 234, 4, 2)，而是{data.shape}")
                    # 尝试重塑数据
                    if len(data.shape) == 3 and data.shape == (234, 4, 2):
                        processed = np.expand_dims(data, axis=0)  # 添加批次维度
                    else:
                        raise ValueError(f"无法处理形状为{data.shape}的数据")
            except Exception as e:
                print(f"加载.npy文件失败: {e}")
                raise
                
        elif file_path.endswith('.bin'):
            # 对于二进制文件，假设它存储的是展平的(234, 4, 2)结构
            print(f"\n处理文件: {os.path.basename(file_path)}")
            print(f"文件大小: {os.path.getsize(file_path)/1024:.2f} KB")
            
            with open(file_path, 'rb') as f:
                byte_data = f.read()
            
            # 将二进制数据转换为numpy数组
            raw_data = np.frombuffer(byte_data, dtype=np.float32)
            
            # 计算元素总数是否满足(234, 4, 2)的要求
            expected_elements = 234 * 4 * 2
            
            if len(raw_data) < expected_elements:
                print(f"警告: 数据元素不足，预期{expected_elements}，实际{len(raw_data)}")
                # 如果数据不足，填充零
                pad_size = expected_elements - len(raw_data)
                raw_data = np.pad(raw_data, (0, pad_size), 'constant')
            
            # 只取一个样本并重塑为(1, 234, 4, 2)
            processed = raw_data[:expected_elements].reshape(1, 234, 4, 2)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，请使用.npy或.bin文件")
        
        # 数据标准化
        mean = np.mean(processed)
        std = np.std(processed)
        std = std if std > 0 else 1e-6  # 防止除零
        processed = (processed - mean) / std
        
        # 数据检查
        print(f"数据范围: [{np.min(processed):.2f}, {np.max(processed):.2f}]")
        print(f"数据均值: {np.mean(processed):.2f} ± {np.std(processed):.2f}")
        
        # 检查并处理NaN和Inf
        if np.isnan(processed).any() or np.isinf(processed).any():
            print("发现NaN或Inf值，已替换")
            processed = np.nan_to_num(processed, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return processed
        
    except Exception as e:
        print(f"\n数据处理失败: {str(e)}")
        print("常见解决方法：")
        print("1. 确保文件为.npy或.bin格式")
        print("2. 确保数据形状符合模型要求(234, 4, 2)")
        print("3. 检查数据采集是否正确")
        print("4. 检查文件是否损坏")
        exit(1)

def preprocess_for_prediction(vmatrix_path, time_steps=10):
    """
    将多个数据包的V矩阵预处理为模型所需的时间序列
    
    参数:
    - vmatrix_path: V矩阵文件路径
    - time_steps: 模型期望的时间步长
    
    返回:
    - 处理好的输入数据
    """
    try:
        # 加载V矩阵数据
        data = np.load(vmatrix_path, allow_pickle=True)
        print(f"原始数据形状: {data.shape}")
        
        # 检查数据类型和结构
        if isinstance(data, np.ndarray) and len(data.shape) == 4 and data.shape[1:] == (234, 4, 2):
            # 数据结构正确: (n_packets, 234, 4, 2)
            packets = data
        elif isinstance(data, list) and len(data) > 0:
            # 如果数据是列表，将其转换为数组
            packets = np.array(data)
            print(f"将列表转换为数组, 形状: {packets.shape}")
        else:
            raise ValueError(f"无法解析数据结构: {type(data)}")
        
        # 创建时间序列
        if len(packets) >= time_steps:
            # 如果有足够的数据包，使用最近的time_steps个
            sequence = packets[-time_steps:]
        else:
            # 如果数据包不足，通过复制数据进行填充
            # 优先使用可用的真实数据
            padding_needed = time_steps - len(packets)
            if len(packets) > 0:
                # 重复第一个包来填充
                padding = np.tile(packets[0:1], (padding_needed, 1, 1, 1))
                sequence = np.concatenate([padding, packets], axis=0)
            else:
                # 如果没有可用数据，创建全零序列
                print("警告: 没有可用数据包，创建零序列")
                sequence = np.zeros((time_steps, 234, 4, 2))
        
        # 确保形状正确
        assert sequence.shape == (time_steps, 234, 4, 2), f"序列形状错误: {sequence.shape}"
        
        # 数据归一化 - 对整个序列进行
        mean = np.mean(sequence)
        std = np.std(sequence)
        std = std if std > 0 else 1e-6  # 防止除以零
        sequence = (sequence - mean) / std
        
        # 添加批次维度
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        print(f"处理后数据形状: {sequence_batch.shape}")
        return sequence_batch
        
    except Exception as e:
        print(f"预处理数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_vmatrix_file(model, vmatrix_path):
    """
    预测V矩阵文件并输出结果
    """
    print(f"\n处理文件: {os.path.basename(vmatrix_path)}")
    
    # 预处理数据
    input_data = preprocess_for_prediction(vmatrix_path)
    
    if input_data is None:
        print("无法处理输入数据")
        return None
    
    # 执行预测
    try:
        predictions = model.predict(input_data, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        
        # 应用softmax确保概率总和为1
        predictions = softmax(predictions)
        confidence = predictions[predicted_class] * 100
        
        # 输出结果
        print("\n预测结果:")
        print(f"预测类别: {label_mapping.get(predicted_class, f'未知({predicted_class})')}")
        print(f"置信度: {confidence:.2f}%")
        print("概率分布:")
        for i, prob in enumerate(predictions):
            label = label_mapping.get(i, f"类别{i}")
            print(f"  {label}: {prob * 100:.2f}%")
            
        return {
            "class": predicted_class,
            "label": label_mapping.get(predicted_class, f'未知({predicted_class})'),
            "confidence": confidence,
            "probabilities": {label_mapping.get(i, f"类别{i}"): prob * 100 for i, prob in enumerate(predictions)}
        }
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

def predict_single_file(model, file_path):
    """
    执行预测并输出结果
    """
    # 数据预处理
    input_data = load_and_preprocess_3d(file_path)
    
    # 检查模型输入形状
    expected_shape = model.input_shape[1:]
    actual_shape = input_data.shape[1:]
    
    print(f"模型期望的输入形状: {expected_shape}")
    print(f"实际数据形状: {actual_shape}")
    
    if expected_shape != actual_shape:
        print("警告: 输入数据形状与模型期望不匹配")
        # 尝试自动调整形状
        if len(expected_shape) == 3 and expected_shape[0] == 234 and expected_shape[2] == 15:
            # 旧模型期望(234, N, 15)格式
            # 将(234, 4, 2)重塑为(234, N, 15)
            print("正在将数据从(234, 4, 2)重塑为", expected_shape)
            input_data = input_data.reshape(1, 234, -1)
            if input_data.shape[2] < expected_shape[1]:
                # 填充
                padded = np.zeros((1, 234, expected_shape[1]))
                padded[:, :, :input_data.shape[2]] = input_data
                input_data = padded
        elif len(expected_shape) == 2 and expected_shape[0] == 234:
            # 将(234, 4, 2)重塑为(234, N)
            print("正在将数据从(234, 4, 2)重塑为", expected_shape)
            input_data = input_data.reshape(1, 234, -1)
    
    # 执行预测
    try:
        predictions = model.predict(input_data)[0]
    except Exception as e:
        print(f"预测失败: {e}")
        print("尝试调整输入数据形状...")
        
        # 如果预测失败，尝试调整输入形状
        if len(model.input_shape) == 4:  # 3D CNN模型
            # 如果模型是3D CNN，添加通道维度
            input_data = np.expand_dims(input_data, axis=-1)
            print(f"添加通道维度后的形状: {input_data.shape}")
            predictions = model.predict(input_data)[0]
        else:
            raise

    # 归一化概率
    predictions = softmax(predictions)

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
        label = label_mapping.get(i, f"类别{i}")
        print(f"  {label}: {prob * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description='CSI动作识别预测')
    parser.add_argument('model_path', type=str, help='模型文件路径 (.h5)')
    parser.add_argument('data_path', type=str, help='要预测的数据文件路径 (.npy或.bin)')
    args = parser.parse_args()

    # 加载模型
    try:
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={
                'ConvNormalization': ConvNormalization,
                'ConvNormalization3D': ConvNormalization3D
            }
        )
        print(f"模型加载成功: {args.model_path}")
        print(f"模型输入形状: {model.input_shape}")
        print(f"模型输出形状: {model.output_shape}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 执行预测
    predict_vmatrix_file(model, args.data_path)