import tensorflow as tf
import numpy as np
import os
import datetime

# 自定义模型层
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

# 手势标签映射
label_mapping = {
    0: 'empty',
    1: 'leftright',
    2: 'updown'
}

def softmax(x):
    """计算softmax以确保正常的概率输出"""
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x)

class GestureModel:
    def __init__(self, model_path, time_steps=10):
        # 添加标签映射作为类属性
        self.label_mapping = {
            0: 'empty',
            1: 'leftright',
            2: 'updown'
        }
        
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        self.model_input_shape = None
        self.time_steps = time_steps  # 添加时间步长参数
        
        if model_path:
            self.load_model(model_path)


            
    def load_model(self, model_path):
        """加载手势识别模型"""
        try:
            # 首先尝试使用标准加载方式
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'ConvNormalization': ConvNormalization,
                    'ConvNormalization3D': ConvNormalization3D
                }
            )
            
            # 保存模型输入形状信息
            self.model_input_shape = self.model.input_shape[1:]
            
            print(f"Model loaded successfully: {os.path.basename(model_path)}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def load_and_preprocess_data(self, file_path, verbose=False):
        """加载并预处理数据以匹配模型输入"""
        try:
            if verbose:
                print(f"\nProcessing file: {os.path.basename(file_path)}")
                print(f"File size: {os.path.getsize(file_path)/1024:.2f} KB")
            
            # 检查文件格式
            if file_path.endswith('.npy'):
                # 直接加载.npy文件
                data = np.load(file_path, allow_pickle=True)
                
                if verbose:
                    print(f"Data shape: {data.shape}")
                
                # 根据输入数据形状进行处理
                if len(data.shape) == 4 and data.shape[1:] == (234, 4, 2):
                    # 取前time_steps个样本或者复制不足的样本，形成时间序列
                    if data.shape[0] >= self.time_steps:
                        processed = data[:self.time_steps]
                    else:
                        # 如果样本不足，则通过复制扩展
                        repeats = int(np.ceil(self.time_steps / data.shape[0]))
                        processed = np.tile(data, (repeats, 1, 1, 1))[:self.time_steps]
                elif len(data.shape) == 3 and data.shape == (234, 4, 2):
                    # 只有一个样本，复制构建时间序列
                    processed = np.tile(np.expand_dims(data, axis=0), (self.time_steps, 1, 1, 1))
                else:
                    raise ValueError(f"Unexpected data shape: {data.shape}")
                    
            elif file_path.endswith('.bin'):
                # 对于二进制文件，假设它们存储扁平化的(234, 4, 2)结构
                with open(file_path, 'rb') as f:
                    byte_data = f.read()
                
                # 将二进制数据转换为numpy数组
                raw_data = np.frombuffer(byte_data, dtype=np.float32)
                
                # 计算总元素是否满足(234, 4, 2)的要求
                expected_elements = 234 * 4 * 2
                
                if len(raw_data) < expected_elements:
                    if verbose:
                        print(f"Warning: Insufficient data elements, expected {expected_elements}, actual {len(raw_data)}")
                    # 如果数据不足，用零填充
                    pad_size = expected_elements - len(raw_data)
                    raw_data = np.pad(raw_data, (0, pad_size), 'constant')
                
                # 重塑为(234, 4, 2)
                single_sample = raw_data[:expected_elements].reshape(234, 4, 2)
                # 复制构建时间序列
                processed = np.tile(np.expand_dims(single_sample, axis=0), (self.time_steps, 1, 1, 1))
            else:
                raise ValueError(f"Unsupported file format: {file_path}, please use .npy or .bin files")
            
            # 数据归一化
            mean = np.mean(processed)
            std = np.std(processed)
            std = std if std > 0 else 1e-6  # 防止除以零
            processed = (processed - mean) / std
            
            if verbose:
                print(f"Processed data shape: {processed.shape}")
                print(f"Data range: [{np.min(processed):.2f}, {np.max(processed):.2f}]")
                print(f"Data mean: {np.mean(processed):.2f} ± {np.std(processed):.2f}")
            
            # 检查并处理NaN和Inf
            if np.isnan(processed).any() or np.isinf(processed).any():
                if verbose:
                    print("Found NaN or Inf values, replaced")
                processed = np.nan_to_num(processed, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 添加批次维度
            processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            print(f"\nData processing failed: {str(e)}")
            print("Common solutions:")
            print("1. Ensure file is in .npy or .bin format")
            print("2. Ensure data shape meets model requirements (time_steps, 234, 4, 2)")
            print("3. Check data collection is correct")
            print("4. Check if file is corrupted")
            return None
    
    def create_sequence_from_sample(self, sample, verbose=False):
        """从单个样本创建时间序列"""
        # 确保sample的形状是(234, 4, 2)
        if sample.shape != (234, 4, 2):
            raise ValueError(f"Expected sample shape (234, 4, 2), got {sample.shape}")
        
        # 创建时间序列 - 复制同一样本time_steps次
        sequence = np.tile(np.expand_dims(sample, axis=0), (self.time_steps, 1, 1, 1))
        
        if verbose:
            print(f"Created sequence with shape: {sequence.shape}")
        
        return sequence
        
    def predict(self, file_path, verbose=False):
        """预测单个文件并返回结果"""
        if not self.is_loaded:
            print("Model not loaded. Cannot make predictions.")
            return None
            
        try:
            # 数据预处理
            input_data = self.load_and_preprocess_data(file_path, verbose)
            
            if input_data is None:
                return None
                
            # 检查输入数据形状和模型期望的输入形状
            if verbose:
                print(f"Model expected input shape: {self.model_input_shape}")
                print(f"Actual data shape: {input_data.shape[1:]}")
            
            # 执行预测
            try:
                predictions = self.model.predict(input_data, verbose=0)[0]
            except Exception as e:
                print(f"Prediction failed: {e}")
                return None
                    
            # 归一化概率
            predictions = softmax(predictions)
            
            # 解析结果
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100  # 转换为百分比
            
            # 构建结果字典
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = {
                "timestamp": timestamp,
                "file": os.path.basename(file_path),
                "predicted_class": predicted_class,
                "predicted_label": self.label_mapping.get(predicted_class, f'Unknown({predicted_class})'),
                "confidence": confidence,
                "probabilities": {self.label_mapping.get(i, f"Class{i}"): prob * 100 for i, prob in enumerate(predictions)}
            }
            
            # 输出结果
            if verbose:
                print("\n" + "="*40)
                print(f"Time: {timestamp}")
                print(f"File: {os.path.basename(file_path)}")
                print(f"Predicted type: {result['predicted_label']} (ID: {predicted_class})")
                print(f"Confidence: {confidence:.2f}%")
                print("Probability distribution:")
                for i, prob in enumerate(predictions):
                    label = self.label_mapping.get(i, f"Class{i}")
                    print(f"  {label}: {prob * 100:.2f}%")
                print("="*40 + "\n")
            
            return result
            
        except Exception as e:
            print(f"Error in prediction process: {e}")
            return None
            
    def predict_realtime(self, samples, file_name = "realtime_sequence", verbose=False):
        if not self.is_loaded:
            print("Model not loaded. Cannot make predictions.")
            return None
            
        try:
            # 确保有足够的样本
            if verbose:
                print(f"Input samples shape: {samples.shape}")
                
            if len(samples.shape) != 4 or samples.shape[1:] != (234, 4, 2):
                raise ValueError(f"Expected samples shape (n, 234, 4, 2), got {samples.shape}")
                
            # 准备输入数据 - 从最新的样本中选择time_steps个
            if samples.shape[0] >= self.time_steps:
                sequence = samples[-self.time_steps:]
            else:
                # 填充不足的数据
                padding_needed = self.time_steps - samples.shape[0]
                # 复制第一个样本进行填充
                padding = np.tile(samples[0:1], (padding_needed, 1, 1, 1))
                sequence = np.concatenate([padding, samples], axis=0)
            
            # 数据归一化
            mean = np.mean(sequence)
            std = np.std(sequence)
            std = std if std > 0 else 1e-6  # 防止除以零
            sequence = (sequence - mean) / std
            
            # 添加批次维度
            sequence = np.expand_dims(sequence, axis=0)
            
            if verbose:
                print(f"Processed sequence shape: {sequence.shape}")
            
            # 执行预测
            predictions = self.model.predict(sequence, verbose=0)[0]
            predictions = softmax(predictions)
            
            # 解析结果
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            
            # 构建结果
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = {
                "timestamp": timestamp,
                "file": file_name,
                "predicted_class": predicted_class,
                "predicted_label": self.label_mapping.get(predicted_class, f'Unknown({predicted_class})'),
                "confidence": confidence,
                "probabilities": {self.label_mapping.get(i, f"Class{i}"): prob * 100 for i, prob in enumerate(predictions)}
            }
            
            # 输出结果
            if verbose:
                print("\n" + "="*40)
                print(f"Time: {timestamp}")
                print(f"Realtime Prediction")
                print(f"Predicted type: {result['predicted_label']} (ID: {predicted_class})")
                print(f"Confidence: {confidence:.2f}%")
                print("Probability distribution:")
                for i, prob in enumerate(predictions):
                    label = self.label_mapping.get(i, f"Class{i}")
                    print(f"  {label}: {prob * 100:.2f}%")
                print("="*40 + "\n")
            
            return result
            
        except Exception as e:
            print(f"Error in realtime prediction: {e}")
            return None
    # 在GestureModel类中添加新方法
    def predict_with_static_detection(self, file_path, verbose=False):
        """增强预测函数，添加静态手势检测逻辑"""
        # 首先进行正常的数据加载和预处理
        input_data = self.load_and_preprocess_data(file_path, verbose)
        
        if input_data is None:
            return None
            
        # 计算动态特征
        if len(input_data.shape) == 5:  # (batch, time_steps, subcarriers, antennas, components)
            sequence = input_data[0]  # 获取第一个样本序列
        else:
            sequence = input_data  # 单个序列
        
        # 计算帧间差异的平均绝对值
        frame_diffs = np.abs(sequence[1:] - sequence[:-1])
        mean_diff = np.mean(frame_diffs)
        
        # 计算信号能量的方差
        signal_magnitude = np.sqrt(sequence[:, :, :, 0]**2 + sequence[:, :, :, 1]**2)
        energy_per_frame = np.sum(signal_magnitude, axis=(1, 2))
        energy_variance = np.var(energy_per_frame)
        
        # 创建静态指数
        static_index = mean_diff * energy_variance
        
        # 静态检测阈值 - 需要根据实际数据调整
        STATIC_THRESHOLD = 3000  # 示例阈值，需要调整

        print(f"静态指数: {static_index:.6f}, 当前阈值: {STATIC_THRESHOLD}")
        
        # 获取模型预测
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # 应用softmax
        predictions = softmax(predictions)
        
        # 如果静态指数小于阈值，强制预测为empty(类别0)
        is_static = static_index < STATIC_THRESHOLD
        
        if is_static:
            # 强制预测为empty
            predictions = np.zeros_like(predictions)
            predictions[0] = 1.0
            if verbose:
                print(f"静态场景检测: 强制预测为empty (静态指数: {static_index:.6f})")
        else:
            if verbose:
                print(f"动态场景检测: 使用模型预测 (静态指数: {static_index:.6f})")
        
        # 解析结果
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class] * 100
        
        # 构建结果字典
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "timestamp": timestamp,
            "file": os.path.basename(file_path),
            "predicted_class": predicted_class,
            "predicted_label": self.label_mapping.get(predicted_class, f'Unknown({predicted_class})'),
            "confidence": confidence,
            "probabilities": {self.label_mapping.get(i, f"Class{i}"): prob * 100 for i, prob in enumerate(predictions)},
            "static_index": static_index,
            "static_threshold": STATIC_THRESHOLD,
            "is_static_detected": is_static
        }
        
        # 输出结果
        if verbose:
            print("\n" + "="*40)
            print(f"Time: {timestamp}")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Predicted type: {result['predicted_label']} (ID: {predicted_class})")
            print(f"Confidence: {confidence:.2f}%")
            print("Probability distribution:")
            for i, prob in enumerate(predictions):
                label = self.label_mapping.get(i, f"Class{i}")
                print(f"  {label}: {prob * 100:.2f}%")
            print(f"Static Index: {static_index:.6f}")
            print("="*40 + "\n")
        
        return result
