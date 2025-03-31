import tensorflow as tf
import numpy as np
import pickle
import os


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
                           selected_subcarriers_idxs=None):
    """
    创建时间序列数据集，将连续样本组合成序列
    
    参数:
    - name_files, labels, ... : 与create_dataset相同的参数
    - time_steps: 每个序列包含的样本数量
    - overlap: 相邻序列之间的重叠样本数量
    
    返回:
    - 序列数据集, 序列数量, 序列标签
    """
    # 首先使用原始create_dataset获取原始数据
    dataset, num_samples, labels_complete = create_dataset(
        name_files, labels, batch_size, M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False,  # 为了保持样本的连续性，不进行洗牌
        cache_file=cache_file, 
        prefetch=False, repeat=False,  # 我们将在处理序列后再添加这些功能
        start_fraction=start_fraction,
        end_fraction=end_fraction,
        selected_subcarriers_idxs=selected_subcarriers_idxs
    )
    
    # 将数据集中的所有样本提取到内存中
    all_samples = []
    all_labels = []
    
    for samples, batch_labels in dataset:
        for i in range(samples.shape[0]):
            all_samples.append(samples[i].numpy())
            all_labels.append(batch_labels[i].numpy())
    
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)
    
    # 计算序列数量
    stride = time_steps - overlap
    num_sequences = max(0, (len(all_samples) - time_steps) // stride + 1)
    
    # 创建序列数据和标签
    seq_samples = []
    seq_labels = []
    
    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        
        # 获取当前序列的样本和标签
        seq_data = all_samples[start_idx:end_idx]
        # 使用序列中最后一个样本的标签作为序列标签
        seq_label = all_labels[end_idx - 1]
        
        seq_samples.append(seq_data)
        seq_labels.append(seq_label)
    
    seq_samples = np.array(seq_samples)
    seq_labels = np.array(seq_labels)
    
    # 构建TensorFlow数据集
    seq_dataset = tf.data.Dataset.from_tensor_slices((seq_samples, seq_labels))
    
    # 应用洗牌、缓存、预取等操作
    if shuffle:
        seq_dataset = seq_dataset.shuffle(buffer_size=num_sequences)
    
    if cache_file:
        seq_dataset = seq_dataset.cache(cache_file)
    
    if batch_size > 0:
        seq_dataset = seq_dataset.batch(batch_size)
    
    if prefetch:
        seq_dataset = seq_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    if repeat:
        seq_dataset = seq_dataset.repeat()
    
    return seq_dataset, num_sequences, seq_labels
