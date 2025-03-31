import tensorflow as tf
import numpy as np
import pickle
import os
import tqdm


def load_numpy(name_f, start_fraction, end_fraction, selected_subcarriers_idxs=None):
    """Load data from NumPy files"""
    data = None
    start_p = 0
    end_p = 0
    
    if name_f.endswith('npy'):
        # Load .npy file
        data = np.load(name_f, allow_pickle=True)
        print(f"Loading file {name_f}, data shape: {data.shape}")
        
        num_samples = data.shape[0]
        start_p = int(start_fraction * num_samples)
        end_p = int(end_fraction * num_samples)
        
        # Select subset
        data = data[start_p:end_p]
    
    elif name_f.endswith('txt'):
        # Original .txt file handling logic preserved
        with open(name_f, "rb") as fp:
            data = pickle.load(fp)
        num_samples = data.shape[0]
        
        start_p = int(start_fraction * num_samples)
        end_p = int(end_fraction * num_samples)
        data = data[start_p:end_p, ...]
    
    # Select specific subcarriers if needed
    if selected_subcarriers_idxs is not None and data is not None:
        data = data[:, selected_subcarriers_idxs, :, :]
    
    return data, start_p, end_p


def create_dataset(name_files, labels, batch_size, M, tx_antennas_list, N, rx_antennas_list, 
                  shuffle, cache_file, prefetch=True, repeat=True, 
                  start_fraction=0, end_fraction=0.8, selected_subcarriers_idxs=None):
    """
    Create dataset for 3D CNN
    Preserve (234, 4, 2) data structure without flattening
    """
    labels_complete = []
    all_features = []
    total_samples = 0
    
    for idx, (file_name, label) in enumerate(zip(name_files, labels)):
        try:
            # Check file existence
            if not os.path.exists(file_name):
                print(f"Warning: File {file_name} not found, skipping...")
                continue
            
            # Load data
            data, start_p, end_p = load_numpy(file_name, start_fraction, end_fraction, selected_subcarriers_idxs)
            
            if data is None or len(data) == 0:
                print(f"Warning: File {file_name} is empty after loading, skipping...")
                continue
                
            print(f"Processing file {idx+1}/{len(name_files)}: {file_name}, samples: {len(data)}")
            
            # Apply label to each sample
            file_labels = [label] * len(data)
            
            # Add to total dataset
            all_features.extend(data)
            labels_complete.extend(file_labels)
            total_samples += len(data)
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    # Convert to NumPy arrays
    features_array = np.array(all_features)
    labels_array = np.array(labels_complete)
    
    print(f"Total loaded samples: {total_samples}")
    print(f"Feature array shape: {features_array.shape}")
    
    # Check for empty data
    if total_samples == 0:
        raise ValueError("No samples loaded. Check file paths and data structure.")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    
    # Apply data augmentation and optimizations
    if shuffle:
        buffer_size = min(total_samples * 2, 10000)  # Prevent oversized buffers
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    if cache_file:
        dataset = dataset.cache(cache_file)
    
    # Batching
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    if repeat:
        dataset = dataset.repeat()
    
    # Dataset validation
    try:
        sample_data, sample_labels = next(iter(dataset))
        print("\n=== Data validation passed ===")
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample labels shape: {sample_labels.shape}")
        print(f"Total samples: {total_samples}")
        print(f"Batch count: {total_samples // batch_size}")
    except Exception as e:
        print("\n!!! Data validation failed !!!")
        raise RuntimeError(f"Data pipeline validation failed: {str(e)}")
    
    return dataset, total_samples, labels_complete

def create_sequence_dataset(name_files, labels, batch_size, M, tx_antennas_list, N, rx_antennas_list,
                           time_steps=10, overlap=0, shuffle=True, cache_file=None,
                           prefetch=True, repeat=True, start_fraction=0, end_fraction=1,
                           selected_subcarriers_idxs=None, verbose=True):
    """
    创建时间序列数据集，将连续样本组合成序列
    
    参数:
    - name_files: 数据文件路径列表
    - labels: 对应的标签列表
    - batch_size: 最终数据集的批处理大小
    - M, tx_antennas_list, N, rx_antennas_list: 天线相关参数
    - time_steps: 每个序列包含的样本数量
    - overlap: 相邻序列之间的重叠样本数量
    - shuffle: 是否打乱数据集
    - cache_file: 缓存文件路径，None表示不缓存
    - prefetch: 是否预取数据
    - repeat: 是否重复数据集
    - start_fraction, end_fraction: 数据集分割参数
    - selected_subcarriers_idxs: 子载波选择索引
    - verbose: 是否打印详细信息
    
    返回:
    - seq_dataset: 序列化后的TensorFlow数据集
    - num_sequences: 序列总数
    - seq_labels: 序列标签列表
    """
    # 设置内部批处理大小，保证效率的同时尽量减少内存使用
    temp_batch_size = min(256, max(20, batch_size))
    
    try:
        if verbose:
            print("开始加载原始数据...")
        
        # 获取原始数据
        unbatched_dataset, num_samples, labels_complete = create_dataset(
            name_files, labels, temp_batch_size, M, tx_antennas_list, N, rx_antennas_list,
            shuffle=False,  # 保持样本的连续性
            cache_file=None,  # 稍后再缓存
            prefetch=False, repeat=False,
            start_fraction=start_fraction,
            end_fraction=end_fraction,
            selected_subcarriers_idxs=selected_subcarriers_idxs
        )
        
        # 解除批处理
        unbatched_dataset = unbatched_dataset.unbatch()
        
        if verbose:
            print(f"解除批处理完成，开始提取样本...")
        
        # 将所有样本提取到内存并创建序列
        all_samples = []
        all_labels = []
        
        # 直接处理数据
        for sample, label in unbatched_dataset:
            all_samples.append(sample.numpy())
            all_labels.append(label.numpy())
        
        all_samples = np.array(all_samples)
        all_labels = np.array(all_labels)
        
        if verbose:
            print(f"加载了 {len(all_samples)} 个样本，形状为 {all_samples.shape}")
        
        # 检查是否有足够的样本创建序列
        if len(all_samples) < time_steps:
            raise ValueError(f"样本数量 ({len(all_samples)}) 小于时间步长 ({time_steps})，无法创建序列")
        
        # 计算序列数量和创建序列
        stride = time_steps - overlap
        if stride <= 0:
            raise ValueError(f"步长必须为正数，当前时间步长为 {time_steps}，重叠为 {overlap}")
            
        num_sequences = max(0, (len(all_samples) - time_steps) // stride + 1)
        
        if verbose:
            print(f"开始创建 {num_sequences} 个序列...")
        
        seq_samples = []
        seq_labels = []
        
        for i in range(num_sequences):
            start_idx = i * stride
            end_idx = start_idx + time_steps
            seq_data = all_samples[start_idx:end_idx]
            
            # 使用多数投票决定序列标签
            unique_labels, counts = np.unique(all_labels[start_idx:end_idx], return_counts=True)
            seq_label = unique_labels[np.argmax(counts)]
            
            seq_samples.append(seq_data)
            seq_labels.append(seq_label)
        
        seq_samples = np.array(seq_samples)
        seq_labels = np.array(seq_labels)
        
        if verbose:
            print(f"创建了 {num_sequences} 个序列，形状为 {seq_samples.shape}")
        
        # 构建TensorFlow数据集
        seq_dataset = tf.data.Dataset.from_tensor_slices((seq_samples, seq_labels))
        
        if shuffle:
            # 使用更大的buffer_size以获得更好的随机性
            buffer_size = min(10000, len(seq_labels))
            seq_dataset = seq_dataset.shuffle(buffer_size=buffer_size)
        
        if cache_file:
            seq_dataset = seq_dataset.cache(cache_file)
            if verbose:
                print(f"数据集已缓存到 {cache_file}")
        
        if batch_size > 0:
            # drop_remainder=False可以避免丢失数据
            seq_dataset = seq_dataset.batch(batch_size, drop_remainder=False)
            if verbose:
                print(f"应用批处理大小 {batch_size}")
        
        if prefetch:
            seq_dataset = seq_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            if verbose:
                print("启用自动预取")
        
        if repeat:
            seq_dataset = seq_dataset.repeat()
            if verbose:
                print("数据集设置为无限重复")
        
        return seq_dataset, num_sequences, seq_labels.tolist()
    
    except Exception as e:
        print(f"创建序列数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise