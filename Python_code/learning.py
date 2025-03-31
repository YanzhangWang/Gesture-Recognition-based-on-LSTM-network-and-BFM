import argparse
import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle
from dataset_utility import *
# Import custom functions
# Import create_dataset from dataset_utility
from dataset_utility import create_dataset
# Import our 3D network model from network_utility
from network_utility import optimal_3d_network

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
    parser.add_argument('model_type', help='Convolutional or attention model type')
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('scenario', help='Scenario to consider, one of {S1, S2, S3, S4, S4_diff, S5, S6, hyper}')
    args = parser.parse_args()

    prefix = args.prefix
    model_name = args.model_name
    
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

    scenario = args.scenario
    if scenario == 'S1':
        # S1 scenario
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_fraction = [0, 0.64]
        val_fraction = [0.64, 0.8]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fraction = [0.8, 1]

    elif scenario == 'Stest':
        # Test scenario
        pos_train_val = [1, 2, 3]
        train_fraction = [0, 0.64]
        val_fraction = [0.64, 0.8]
        pos_test = [4, 5]
        test_fraction = [0.8, 1]

    elif scenario == 'S2':
        # S2 scenario
        pos_train_val = [1, 3, 5, 7, 9]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [2, 4, 6, 8]
        test_fraction = [0, 1]
    elif scenario == 'S3':
        # S3 scenario
        pos_train_val = [1, 2, 3, 4, 5]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [6, 7, 8, 9]
        test_fraction = [0, 1]
    elif scenario == 'S4':
        # S4 mobility scenario
        pos_train_val = [5, 6, 7, 8]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [9, 10, 11]
        test_fraction = [0, 1]
    elif scenario == 'S4_diff':
        # S4 different mobility scenario
        pos_train_val = [5, 6, 7, 8]
        train_fraction = [0, 0.4]
        val_fraction = [0.4, 0.5]
        pos_test = [9, 10, 11]
        test_fraction = [0.6, 0.8]
    elif scenario == 'S5':
        # S5 mobility scenario
        pos_train_val = [1, 2, 3, 4]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [5, 6, 7, 8, 9, 10, 11]
        test_fraction = [0, 1]
    elif scenario == 'S6':
        # S6 mobility scenario
        pos_train_val = [5, 6, 7, 8, 9, 10, 11]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [1, 2, 3, 4]
        test_fraction = [0, 1]
    elif scenario == 'hyper':
        # Hyperparameter selection scenario
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_fraction = [0, 0.5]
        val_fraction = [0.5, 0.8]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fraction = [0.8, 1]

    # Positions and device IDs
    num_pos = args.positions
    extension = '.npy'  
    module_IDs = ['vmatrices_updown', 'vmatrices_leftright', 'vmatrices_stand', 'vmatrices_empty']

    # TX and RX antennas selection
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

    # Subcarriers selection
    selected_subcarriers_idxs = None  # default (80 MHz)
    num_selected_subcarriers = 234
    bandwidth = args.bandwidth

    label_to_index = {'vmatrices_updown': 0, 'vmatrices_leftright': 1, 'vmatrices_stand': 2, 'vmatrices_empty': 3}

    name_files_train = []
    labels_train = []
    name_files_val = []
    labels_val = []
    name_files_test = []
    labels_test = []

    input_dir = args.dir + '/'

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
    
    #
    name_cache_train = './cache_files/' + model_name + 'cache_train'
    dataset_train, num_samples_train, labels_complete_train = create_dataset(
        name_files_train, labels_train, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=True, cache_file=name_cache_train,
        prefetch=True, repeat=True,
        start_fraction=train_fraction[0],
        end_fraction=train_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)
    #

    time_steps = 10  # 可以根据需要调整
    overlap = 5      # 可以根据需要调整
    name_cache_train = './cache_files/' + model_name + 'cache_train_seq'
    dataset_train, num_samples_train, labels_complete_train = create_sequence_dataset(
    name_files_train, labels_train, batch_size,
    M, tx_antennas_list, N, rx_antennas_list,
    time_steps=time_steps, overlap=overlap,
    shuffle=True, cache_file=name_cache_train,
    prefetch=True, repeat=True,
    start_fraction=train_fraction[0],
    end_fraction=train_fraction[1],
    selected_subcarriers_idxs=selected_subcarriers_idxs)

    #
    name_cache_val = './cache_files/' + model_name + 'cache_val'
    dataset_val, num_samples_val, labels_complete_val = create_dataset(
        name_files_val, labels_val, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False, cache_file=name_cache_val,
        prefetch=True, repeat=True,
        start_fraction=val_fraction[0],
        end_fraction=val_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)
    #

    name_cache_val = './cache_files/' + model_name + 'cache_train_seq'
    dataset_val, num_samples_val, labels_complete_val = create_sequence_dataset(
    name_files_val, labels_val, batch_size,
    M, tx_antennas_list, N, rx_antennas_list,
    time_steps=time_steps, overlap=overlap,
    shuffle=True, cache_file=name_cache_val,
    prefetch=True, repeat=True,
    start_fraction=val_fraction[0],
    end_fraction=val_fraction[1],
    selected_subcarriers_idxs=selected_subcarriers_idxs)

    #
    name_cache_test = './cache_files/' + model_name + 'cache_test'
    dataset_test, num_samples_test, labels_complete_test = create_dataset(
        name_files_test, labels_test, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False, cache_file=name_cache_test,
        prefetch=True, repeat=True,
        start_fraction=test_fraction[0],
        end_fraction=test_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)
    #


    name_cache_test = './cache_files/' + model_name + 'cache_train_seq'
    dataset_test, num_samples_test, labels_complete_test = create_sequence_dataset(
    name_files_test, labels_test, batch_size,
    M, tx_antennas_list, N, rx_antennas_list,
    time_steps=time_steps, overlap=overlap,
    shuffle=True, cache_file=name_cache_test,
    prefetch=True, repeat=True,
    start_fraction=test_fraction[0],
    end_fraction=test_fraction[1],
    selected_subcarriers_idxs=selected_subcarriers_idxs)

    
    input_shape = (num_selected_subcarriers, 4, 2)  
    print(f"Input shape: {input_shape}")
    
    num_classes = len(module_IDs)
    print(f"Number of classes: {num_classes}")

    # Use 3D network model
    # model_net = optimal_3d_network(input_shape, num_classes)
    model_net = optimal_2d_sequence_network(input_shape, num_classes, time_steps=time_steps)
    optimiz = tf.keras.optimizers.Adam(learning_rate=1E-4)  # Adjust learning rate appropriately

    model_net.summary()

    # Training setup
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_net.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    train_steps_per_epoch = int(np.ceil(num_samples_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val / batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))

    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    name_save = model_name + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                'hybrid_cnn_lstm_model'  # Flag for 3D model

    name_model = './network_models/' + name_save + 'network.h5'

    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                      monitor='val_sparse_categorical_accuracy')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    print("Starting training...")
    results = model_net.fit(dataset_train, epochs=30, steps_per_epoch=train_steps_per_epoch,
                           validation_data=dataset_val, validation_steps=val_steps_per_epoch,
                           callbacks=[callback_stop, callback_save, tensorboard_callback])

    print("Loading best model...")
    best_model = tf.keras.models.load_model(name_model)
    model_net = best_model

    # Testing
    print("Testing model...")
    prediction_test = model_net.predict(dataset_test, steps=test_steps_per_epoch)[:len(labels_complete_test)]

    labels_pred_test = np.argmax(prediction_test, axis=1)

    labels_complete_test_array = np.asarray(labels_complete_test)
    conf_matrix_test = confusion_matrix(labels_complete_test_array, labels_pred_test,
                                       labels=list(range(num_classes)),
                                       normalize='true')
    precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(labels_complete_test_array,
                                                                                 labels_pred_test,
                                                                                 labels=list(range(num_classes)))
    accuracy_test = accuracy_score(labels_complete_test_array, labels_pred_test)
    print('Test accuracy: %.5f' % accuracy_test)

    # Validation
    prediction_val = model_net.predict(dataset_val, steps=val_steps_per_epoch)[:len(labels_complete_val)]

    labels_pred_val = np.argmax(prediction_val, axis=1)

    labels_complete_val_array = np.asarray(labels_complete_val)
    conf_matrix_val = confusion_matrix(labels_complete_val_array, labels_pred_val,
                                      labels=list(range(num_classes)),
                                      normalize='true')
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(labels_complete_val_array,
                                                                             labels_pred_val,
                                                                             labels=list(range(num_classes)))
    accuracy_val = accuracy_score(labels_complete_val_array, labels_pred_val)
    print('Validation accuracy: %.5f' % accuracy_val)

    # Training set evaluation
    name_cache_train_test = './cache_files/' + model_name + 'cache_train_test'
    dataset_train, num_samples_train, labels_complete_train = create_dataset(
        name_files_train, labels_train, batch_size,
        M, tx_antennas_list, N, rx_antennas_list,
        shuffle=False,
        cache_file=name_cache_train_test,
        prefetch=True, repeat=True,
        start_fraction=train_fraction[0],
        end_fraction=train_fraction[1],
        selected_subcarriers_idxs=selected_subcarriers_idxs)

    prediction_train = model_net.predict(dataset_train, steps=train_steps_per_epoch)[:len(labels_complete_train)]

    labels_pred_train = np.argmax(prediction_train, axis=1)

    labels_complete_train_array = np.asarray(labels_complete_train)
    conf_matrix_train_test = confusion_matrix(labels_complete_train_array, labels_pred_train,
                                             labels=list(range(num_classes)),
                                             normalize='true')
    precision_train_test, recall_train_test, fscore_train_test, _ = precision_recall_fscore_support(
        labels_complete_train_array, labels_pred_train, labels=list(range(num_classes)))
    accuracy_train_test = accuracy_score(labels_complete_train_array, labels_pred_train)
    print('Training accuracy: %.5f' % accuracy_train_test)

    trainable_parameters = np.sum([np.prod(v.get_shape()) for v in model_net.trainable_weights])
    metrics_dict = {'trainable_parameters': trainable_parameters,
                   'conf_matrix_train': conf_matrix_train_test, 'accuracy_train': accuracy_train_test,
                   'precision_train': precision_train_test, 'recall_train': recall_train_test,
                   'fscore_train': fscore_train_test,
                   'conf_matrix_val': conf_matrix_val, 'accuracy_val': accuracy_val,
                   'precision_val': precision_val, 'recall_val': recall_val, 'fscore_val': fscore_val,
                   'conf_matrix_test': conf_matrix_test, 'accuracy_test': accuracy_test,
                   'precision_test': precision_test, 'recall_test': recall_test, 'fscore_test': fscore_test
                  }

    name_file = './outputs/' + name_save + '.txt'

    # Create directory if missing
    directory = os.path.dirname(name_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Saving results...")
    with open(name_file, "wb") as fp:
        pickle.dump(metrics_dict, fp)
    
    print("Done!")


