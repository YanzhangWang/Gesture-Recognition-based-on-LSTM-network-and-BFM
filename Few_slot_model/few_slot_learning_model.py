import argparse
import os
import shutil
import numpy as np
import tensorflow as tf
from network_utility import residual_static_gesture_model, optimal_3d_network
from dataset_utility import create_dataset
from network_models import PrototypicalNetwork, create_siamese_network, build_maml_model
from learning_methods import (train_normal_model, train_siamese_network, train_prototype_network, 
                            train_maml, evaluate_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Data directory')
    parser.add_argument('positions', help='Number of distinct positions', type=int)
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('M', help='Number of transmit antennas', type=int)
    parser.add_argument('N', help='Number of receive antennas', type=int)
    parser.add_argument('tx_antennas', help='TX antenna indices to consider (comma-separated)')
    parser.add_argument('rx_antennas', help='RX antenna indices to consider (comma-separated)')
    parser.add_argument('bandwidth', help='Subcarrier bandwidth selection [MHz], can be 20, 40, 80 (default 80)', type=int)
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('scenario', help='Scenario to consider, one of {S1, S2, S3, S4, S4_diff, S5, S6, hyper}')
    parser.add_argument('--method', help='Learning method: normal, siamese, prototype, or maml', default='normal')
    parser.add_argument('--n_way', help='Number of classes for few-shot learning', type=int, default=3)
    parser.add_argument('--k_shot', help='Number of samples per class for few-shot learning', type=int, default=5)
    args = parser.parse_args()

    prefix = args.prefix
    model_name = args.model_name
    method = args.method  # New parameter: learning method
    n_way = args.n_way    # New parameter: number of classes
    k_shot = args.k_shot  # New parameter: samples per class
    
    # Create necessary directories
    os.makedirs('./cache_files/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    os.makedirs('./network_models/', exist_ok=True)
    os.makedirs('./outputs/', exist_ok=True)
    
    # Remove existing cache files
    if os.path.exists('./cache_files/'):
        list_cache_files = os.listdir('./cache_files/')
        for file_cache in list_cache_files:
            if file_cache.startswith(model_name):
                os.remove('./cache_files/' + file_cache)
    
    if os.path.exists('./logs/train/'):
        shutil.rmtree('./logs/train/')
    if os.path.exists('./logs/validation/'):
        shutil.rmtree('./logs/validation/')

    # Scenario setup
    scenario = args.scenario
    if scenario == 'S1':
        # S1 scenario
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_fraction = [0, 0.64]
        val_fraction = [0.64, 0.8]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fraction = [0.8, 1]

    # Position and device ID
    num_pos = args.positions
    extension = '.npy'  
    module_IDs = ['vmatrices_empty', 'vmatrices_stop', 'vmatrices_first']

    # TX and RX antenna selection
    M = args.M
    N = args.N
    tx_antennas = args.tx_antennas
    tx_antennas_list = []
    for lab_act in tx_antennas.split(','):
        lab_act = int(lab_act)
        if lab_act >= M:
            print('Error in tx_antennas input argument')
            break
        tx_antennas_list.append(lab_act)

    rx_antennas = args.rx_antennas
    rx_antennas_list = []
    for lab_act in rx_antennas.split(','):
        lab_act = int(lab_act)
        if lab_act >= N:
            print('Error in rx_antennas input argument')
            break
        rx_antennas_list.append(lab_act)

    # Subcarrier selection
    selected_subcarriers_idxs = None  # default (80 MHz)
    num_selected_subcarriers = 234
    bandwidth = args.bandwidth

    label_to_index = {'vmatrices_empty': 0, 'vmatrices_stop': 1, 'vmatrices_first': 2}

    # Generate file names and label lists
    name_files_train = []
    labels_train = []
    name_files_val = []
    labels_val = []
    name_files_test = []
    labels_test = []

    input_dir = args.dir + '/'

    # Generate file and label lists code
    for mod_ID in module_IDs:
        for pos in pos_train_val:
            pos_id = pos 
            if pos_id == 10:
                pos_id = 'A'
            name_file = input_dir + mod_ID + '_' + str(pos_id) + extension
            name_files_train.append(name_file)
            labels_train.append(label_to_index[mod_ID]) 

    for mod_ID in module_IDs:
        for pos in pos_train_val:
            pos_id = pos 
            if pos_id == 10:
                pos_id = 'A'
            name_file = input_dir + mod_ID + '_' + str(pos_id) + extension
            name_files_val.append(name_file)
            labels_val.append(label_to_index[mod_ID])

    for mod_ID in module_IDs:
        for pos in pos_test:
            pos_id = pos 
            if pos_id == 10:
                pos_id = 'A'
            name_file = input_dir + mod_ID + '_' + str(pos_id) + extension
            name_files_test.append(name_file)
            labels_test.append(label_to_index[mod_ID])

    batch_size = 32
    
    # Create datasets
    name_cache_train = './cache_files/' + model_name + 'cache_train'
    dataset_train, num_samples_train, labels_complete_train = create_dataset(
        name_files_train, labels_train, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=True, cache_file=name_cache_train,
        prefetch=True, repeat=True,
        start_fraction=train_fraction[0],
        end_fraction=train_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)

    name_cache_val = './cache_files/' + model_name + 'cache_val'
    dataset_val, num_samples_val, labels_complete_val = create_dataset(
        name_files_val, labels_val, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False, cache_file=name_cache_val,
        prefetch=True, repeat=True,
        start_fraction=val_fraction[0],
        end_fraction=val_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)

    name_cache_test = './cache_files/' + model_name + 'cache_test'
    dataset_test, num_samples_test, labels_complete_test = create_dataset(
        name_files_test, labels_test, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False, cache_file=name_cache_test,
        prefetch=True, repeat=True,
        start_fraction=test_fraction[0],
        end_fraction=test_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)
    
    train_steps_per_epoch = int(np.ceil(num_samples_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val / batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))
    
    # Extract all samples and labels from dataset to prepare for few-shot learning
    all_samples_train = []
    for x, y in dataset_train.take(train_steps_per_epoch):
        all_samples_train.extend(x.numpy())
    all_labels_train = labels_complete_train
    
    input_shape = (num_selected_subcarriers, 4, 2)  
    print(f"Input shape: {input_shape}")
    
    num_classes = len(module_IDs)
    print(f"Number of classes: {num_classes}")

    # Create model based on selected method
    if method == 'normal':
        # Use standard network model
        model_net = residual_static_gesture_model(input_shape, num_classes)
        
        # Train the model
        model_net, labels_pred_test = train_normal_model(
            model_net, dataset_train, dataset_val, dataset_test,
            train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch,
            labels_complete_test, num_classes, model_name, tx_antennas_list,
            rx_antennas_list, pos_train_val, pos_test, bandwidth
        )
        
    elif method == 'siamese':
        # Use Siamese Network
        model_net = None  # No central model for evaluation metrics
        
        # Train Siamese Network
        labels_pred_test = train_siamese_network(
            dataset_train, dataset_test, input_shape, train_steps_per_epoch, test_steps_per_epoch,
            labels_complete_test, num_classes, k_shot, model_name, tx_antennas_list,
            rx_antennas_list, pos_train_val, pos_test, bandwidth
        )
        
    elif method == 'prototype':
        # Use Prototypical Network
        model_net = None  # No central model for evaluation metrics
        
        # Train Prototypical Network
        labels_pred_test = train_prototype_network(
            dataset_train, dataset_test, input_shape, test_steps_per_epoch,
            labels_complete_test, num_classes, n_way, k_shot, model_name, tx_antennas_list,
            rx_antennas_list, pos_train_val, pos_test, bandwidth
        )
        
    elif method == 'maml':
        # Use MAML
        model_net = None  # No central model for evaluation metrics
        
        # Train MAML
        labels_pred_test = train_maml(
            dataset_train, dataset_test, input_shape, test_steps_per_epoch,
            labels_complete_test, num_classes, n_way, k_shot, model_name, tx_antennas_list,
            rx_antennas_list, pos_train_val, pos_test, bandwidth
        )
    
    else:
        raise ValueError(f"Unsupported learning method: {method}")
    
    # Evaluate model
    if model_net is None:
        # For few-shot methods, create a dummy model for evaluation metrics
        model_net = tf.keras.Sequential()
    
    metrics_dict = evaluate_model(
        labels_pred_test, labels_complete_test, num_classes, method, n_way, k_shot,
        model_net, model_name, tx_antennas_list, rx_antennas_list, pos_train_val,
        pos_test, bandwidth
    )
    
    print("Done!")