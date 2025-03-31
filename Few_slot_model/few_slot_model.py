import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def generate_pairs(features, labels, num_pairs=1000):
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Create a mapping from class to sample indices
    class_indices = {}
    for label in unique_labels:
        class_indices[label] = np.where(labels == label)[0]
    
    pairs_features = []
    pairs_labels = []
    
    # Generate similar and dissimilar pairs
    n_positive = num_pairs // 2
    n_negative = num_pairs - n_positive
    
    # Similar pairs (same class)
    for _ in range(n_positive):
        # Randomly select a class
        class_idx = np.random.choice(unique_labels)
        # Randomly select two samples from this class
        if len(class_indices[class_idx]) >= 2:
            idx1, idx2 = np.random.choice(class_indices[class_idx], 2, replace=False)
            pairs_features.append([features[idx1], features[idx2]])
            pairs_labels.append(1)  # Same class
    
    # Dissimilar pairs (different classes)
    for _ in range(n_negative):
        # Randomly select two different classes
        if n_classes >= 2:
            class1, class2 = np.random.choice(unique_labels, 2, replace=False)
            # Randomly select one sample from each class
            idx1 = np.random.choice(class_indices[class1])
            idx2 = np.random.choice(class_indices[class2])
            pairs_features.append([features[idx1], features[idx2]])
            pairs_labels.append(0)  # Different classes
    
    return np.array(pairs_features), np.array(pairs_labels)

def create_episode(features, labels, n_way=5, k_shot=1, n_query=15):
    unique_classes = np.unique(labels)
    
    if len(unique_classes) < n_way:
        raise ValueError(f"Not enough classes: need {n_way}, but only have {len(unique_classes)}")
    
    # Randomly select n_way classes
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)
    
    support_x = []
    support_y = []
    query_x = []
    query_y = []
    
    for i, cls in enumerate(selected_classes):
        # Find samples belonging to current class
        cls_indices = np.where(labels == cls)[0]
        
        if len(cls_indices) < k_shot + n_query:
            # If not enough samples, use oversampling
            selected_indices = np.random.choice(cls_indices, k_shot + n_query, replace=True)
        else:
            # Randomly select k_shot + n_query samples
            selected_indices = np.random.choice(cls_indices, k_shot + n_query, replace=False)
        
        # Split into support set and query set
        support_indices = selected_indices[:k_shot]
        query_indices = selected_indices[k_shot:k_shot + n_query]
        
        # Collect support set
        for idx in support_indices:
            support_x.append(features[idx])
            support_y.append(i)  # Use label from 0 to n_way-1
        
        # Collect query set
        for idx in query_indices:
            query_x.append(features[idx])
            query_y.append(i)
    
    # Convert to numpy arrays
    support_x = np.array(support_x)
    support_y = np.array(support_y)
    query_x = np.array(query_x)
    query_y = np.array(query_y)
    
    return support_x, support_y, query_x, query_y

def train_normal_model(model_net, dataset_train, dataset_val, dataset_test, 
                      train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch, 
                      labels_complete_test, num_classes, model_name, tx_antennas_list, 
                      rx_antennas_list, pos_train_val, pos_test, bandwidth):
    # Compile model
    optimiz = tf.keras.optimizers.Adam(learning_rate=1E-4)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_net.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    # Set up callbacks
    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    name_save = model_name + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                'residual_static_gesture_model'
    
    name_model = './network_models/' + name_save + 'network.h5'
    
    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                     monitor='val_sparse_categorical_accuracy')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    
    # Start training
    print("Starting training...")
    results = model_net.fit(dataset_train, epochs=30, steps_per_epoch=train_steps_per_epoch,
                           validation_data=dataset_val, validation_steps=val_steps_per_epoch,
                           callbacks=[callback_stop, callback_save, tensorboard_callback])
    
    # Load best model
    print("Loading best model...")
    best_model = tf.keras.models.load_model(name_model)
    model_net = best_model
    
    # Test
    print("Testing model...")
    prediction_test = model_net.predict(dataset_test, steps=test_steps_per_epoch)[:len(labels_complete_test)]
    labels_pred_test = np.argmax(prediction_test, axis=1)
    
    return model_net, labels_pred_test

def train_siamese_network(dataset_train, dataset_test, input_shape, train_steps_per_epoch, test_steps_per_epoch,
                         labels_complete_test, num_classes, k_shot, model_name, tx_antennas_list, 
                         rx_antennas_list, pos_train_val, pos_test, bandwidth):
    from network_models import create_siamese_network, contrastive_loss
    
    print("Using Siamese Network for few-shot learning")
    
    # Collect training samples
    print("Collecting training samples...")
    all_samples_train = []
    all_labels_train = []
    
    # Collect a certain number of batches
    for samples, labels in dataset_train.take(50):  # Collect 50 batches of data
        all_samples_train.extend(samples.numpy())
        all_labels_train.extend(labels.numpy())
    
    all_samples_train = np.array(all_samples_train)
    all_labels_train = np.array(all_labels_train)
    
    # Generate training pairs
    print("Generating training pairs...")
    train_pairs, train_pair_labels = generate_pairs(all_samples_train, all_labels_train, num_pairs=5000)
    
    # Create and compile siamese network
    siamese_model, feature_extractor = create_siamese_network(input_shape)
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=1.0)
    )
    
    # Train siamese network
    print("Training Siamese Network...")
    
    batch_size = 32
    epochs = 30
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle data
        indices = np.random.permutation(len(train_pair_labels))
        shuffled_pairs = train_pairs[indices]
        shuffled_labels = train_pair_labels[indices]
        
        # Train in batches
        num_batches = len(indices) // batch_size
        epoch_loss = 0
        epoch_acc = 0
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size
            
            batch_pairs = shuffled_pairs[start_idx:end_idx]
            batch_labels = shuffled_labels[start_idx:end_idx]
            
            # Train one step
            loss = siamese_model.train_on_batch(
                [batch_pairs[:, 0], batch_pairs[:, 1]], 
                batch_labels
            )
            
            # Calculate accuracy
            predictions = siamese_model.predict([batch_pairs[:, 0], batch_pairs[:, 1]])
            pred_classes = (predictions < 0.5).astype(int).flatten()
            accuracy = np.mean(pred_classes == batch_labels)
            
            epoch_loss += loss
            epoch_acc += accuracy
        
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        
        print(f"loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")
    
    # Save model
    name_save = model_name + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_siamese'
    
    feature_extractor.save('./network_models/' + name_save + '_feature_extractor.h5')
    siamese_model.save('./network_models/' + name_save + '_siamese.h5')
    
    # Test - using few-shot classification
    print("Testing with few-shot classification...")
    
    # Collect test samples
    all_samples_test = []
    all_labels_test = []
    
    print(f"Collecting complete test set, total {test_steps_per_epoch} batches...")
    for samples, labels in dataset_test.take(test_steps_per_epoch):
        all_samples_test.extend(samples.numpy())
        all_labels_test.extend(labels.numpy())

    all_samples_test = np.array(all_samples_test)
    all_labels_test = np.array(all_labels_test)

    print(f"Number of collected test samples: {len(all_samples_test)}")
    print(f"Number of complete test labels: {len(labels_complete_test)}")

    # Ensure collected test samples match complete test labels length
    if len(all_samples_test) > len(labels_complete_test):
        # Truncate excess samples
        all_samples_test = all_samples_test[:len(labels_complete_test)]
        all_labels_test = all_labels_test[:len(labels_complete_test)]
    elif len(all_samples_test) < len(labels_complete_test):
        # Use existing test samples and corresponding labels
        labels_complete_test = labels_complete_test[:len(all_samples_test)]
    
    # Select k samples for each class as support set
    support_set_x = []
    support_set_y = []
    
    for class_idx in range(num_classes):
        # Find samples belonging to current class
        class_samples = all_samples_train[all_labels_train == class_idx]
        if len(class_samples) >= k_shot:
            # Randomly select k samples
            selected_samples = class_samples[:k_shot]
            support_set_x.extend(selected_samples)
            support_set_y.extend([class_idx] * k_shot)
    
    support_set_x = np.array(support_set_x)
    support_set_y = np.array(support_set_y)
    
    # Use feature extractor to get support set features
    support_features = feature_extractor.predict(support_set_x)
    
    # Evaluate test set
    correct = 0
    total = 0
    
    # Prediction results list
    labels_pred_test = []
    
    for i in range(len(all_samples_test)):
        # Extract features
        query_feature = feature_extractor.predict(np.expand_dims(all_samples_test[i], axis=0))[0]
        
        # Calculate distance to each sample in support set
        distances = []
        for support_feature in support_features:
            distance = np.sqrt(np.sum((query_feature - support_feature) ** 2))
            distances.append(distance)
        
        # Predict as class with minimum distance
        predicted_class = support_set_y[np.argmin(distances)]
        true_class = all_labels_test[i]
        
        # Save prediction result
        labels_pred_test.append(predicted_class)
        
        if predicted_class == true_class:
            correct += 1
        total += 1
    
    # Calculate total accuracy
    accuracy_test = correct / total if total > 0 else 0
    print(f"Few-shot learning test accuracy: {accuracy_test:.4f}")
    
    # Convert to numpy array
    labels_pred_test = np.array(labels_pred_test)
    
    return labels_pred_test

def train_prototype_network(dataset_train, dataset_test, input_shape, test_steps_per_epoch,
                           labels_complete_test, num_classes, n_way, k_shot, model_name, tx_antennas_list, 
                           rx_antennas_list, pos_train_val, pos_test, bandwidth):
    from network_models import PrototypicalNetwork
    
    print("Using Prototypical Network for few-shot learning")
    
    # Collect training and test samples
    print("Collecting samples...")
    all_samples_train = []
    all_labels_train = []
    
    for samples, labels in dataset_train.take(50):  # Collect 50 batches of data
        all_samples_train.extend(samples.numpy())
        all_labels_train.extend(labels.numpy())
    
    all_samples_train = np.array(all_samples_train)
    all_labels_train = np.array(all_labels_train)
    
    all_samples_test = []
    all_labels_test = []
    
    for samples, labels in dataset_test.take(50):
        all_samples_test.extend(samples.numpy())
        all_labels_test.extend(labels.numpy())
    
    all_samples_test = np.array(all_samples_test)
    all_labels_test = np.array(all_labels_test)
    
    # Create and compile prototypical network
    proto_model = PrototypicalNetwork(input_shape)
    proto_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Train prototypical network
    print("Training Prototypical Network...")
    
    epochs = 200  # Meta-learning typically requires more training epochs
    n_way = min(num_classes, n_way)  # Ensure n_way doesn't exceed actual number of classes
    n_query = 15  # Number of query samples per class
    
    for episode in range(epochs):
        # Create a training episode
        support_x, support_y, query_x, query_y = create_episode(
            all_samples_train, all_labels_train, n_way=n_way, k_shot=k_shot, n_query=n_query
        )
        
        # Train one step
        with tf.GradientTape() as tape:
            # Forward pass
            logits = proto_model(support_x, support_y, query_x, n_way, k_shot, training=True)
            
            # Calculate loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                query_y, logits, from_logits=True
            )
            loss = tf.reduce_mean(loss)
        
        # Backpropagation
        gradients = tape.gradient(loss, proto_model.trainable_variables)
        proto_model.optimizer.apply_gradients(zip(gradients, proto_model.trainable_variables))
        
        # Calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{epochs}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")
    
    # Save model
    name_save = model_name + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_prototype'
    
    proto_model.save_weights('./network_models/' + name_save + '_prototype.h5')
    
    # Test prototypical network
    print("Testing Prototypical Network...")
    
    # Select k samples for each class as support set
    support_set_x = []
    support_set_y = []
    
    for class_idx in range(num_classes):
        # Find samples belonging to current class
        class_samples = all_samples_train[all_labels_train == class_idx]
        if len(class_samples) >= k_shot:
            # Select first k samples
            selected_samples = class_samples[:k_shot]
            support_set_x.extend(selected_samples)
            support_set_y.extend([class_idx] * k_shot)
    
    support_set_x = np.array(support_set_x)
    support_set_y = np.array(support_set_y)
    
    # Evaluate test set
    total_accuracy = 0
    num_batches = 0
    
    # Prediction results list
    labels_pred_test = []
    
    # Process test samples in batches due to potentially large number
    batch_size = 100
    for start_idx in range(0, len(all_samples_test), batch_size):
        end_idx = min(start_idx + batch_size, len(all_samples_test))
        batch_samples = all_samples_test[start_idx:end_idx]
        batch_labels = all_labels_test[start_idx:end_idx]
        
        # Use prototypical network for prediction
        logits = proto_model(support_set_x, support_set_y, batch_samples, num_classes, k_shot, training=False)
        predictions = tf.argmax(logits, axis=1).numpy()
        
        # Add to prediction results list
        labels_pred_test.extend(predictions)
        
        # Calculate accuracy
        batch_accuracy = np.mean(predictions == batch_labels)
        total_accuracy += batch_accuracy
        num_batches += 1
    
    # Calculate total accuracy
    accuracy_test = total_accuracy / num_batches if num_batches > 0 else 0
    print(f"Few-shot learning test accuracy: {accuracy_test:.4f}")
    
    # Convert to numpy array
    labels_pred_test = np.array(labels_pred_test)
    
    return labels_pred_test

def train_maml(dataset_train, dataset_test, input_shape, test_steps_per_epoch,
              labels_complete_test, num_classes, n_way, k_shot, model_name, tx_antennas_list, 
              rx_antennas_list, pos_train_val, pos_test, bandwidth):
    from network_models import build_maml_model
    
    print("Using MAML for few-shot learning")
    
    # Collect training samples
    print("Collecting samples...")
    all_samples_train = []
    all_labels_train = []
    
    for samples, labels in dataset_train.take(50):
        all_samples_train.extend(samples.numpy())
        all_labels_train.extend(labels.numpy())
    
    all_samples_train = np.array(all_samples_train)
    all_labels_train = np.array(all_labels_train)
    
    all_samples_test = []
    all_labels_test = []
    
    for samples, labels in dataset_test.take(50):
        all_samples_test.extend(samples.numpy())
        all_labels_test.extend(labels.numpy())
    
    all_samples_test = np.array(all_samples_test)
    all_labels_test = np.array(all_labels_test)
    
    # Create MAML meta-model
    meta_model = build_maml_model(input_shape, num_classes)
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # MAML parameters
    inner_lr = 0.01
    meta_epochs = 100
    n_way = min(num_classes, n_way)
    n_query = 15
    tasks_per_batch = 4
    
    # Train MAML
    print("Training MAML...")
    
    for epoch in range(meta_epochs):
        meta_loss = 0
        meta_accuracy = 0
        
        # Each batch contains multiple tasks
        for _ in range(tasks_per_batch):
            # Create a task (episode)
            support_x, support_y, query_x, query_y = create_episode(
                all_samples_train, all_labels_train, n_way=n_way, k_shot=k_shot, n_query=n_query
            )
            
            # Create task-specific model
# Create task-specific model
            task_model = build_maml_model(input_shape, num_classes)
            task_model.set_weights(meta_model.get_weights())
            
            # Inner loop optimization (fast adaptation on support set)
            with tf.GradientTape() as inner_tape:
                support_logits = task_model(support_x, training=True)
                support_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    support_y, support_logits, from_logits=True
                )
                support_loss = tf.reduce_mean(support_loss)
            
            # Calculate inner loop gradients
            inner_grads = inner_tape.gradient(support_loss, task_model.trainable_variables)
            
            # Manually update task model
            updated_vars = []
            for var, grad in zip(task_model.trainable_variables, inner_grads):
                updated_vars.append(var - inner_lr * grad)
            
            task_model.set_weights(updated_vars)
            
            # Evaluate adapted model on query set
            with tf.GradientTape() as outer_tape:
                query_logits = task_model(query_x, training=True)
                query_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    query_y, query_logits, from_logits=True
                )
                query_loss = tf.reduce_mean(query_loss)
            
            # Calculate outer loop gradients
            meta_grads = outer_tape.gradient(query_loss, meta_model.trainable_variables)
            
            # Update meta-model
            meta_optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))
            
            # Evaluate performance
            predictions = tf.argmax(query_logits, axis=1)
            task_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
            )
            
            meta_loss += query_loss
            meta_accuracy += task_accuracy
        
        # Calculate average performance
        meta_loss /= tasks_per_batch
        meta_accuracy /= tasks_per_batch
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{meta_epochs}, Meta Loss: {meta_loss.numpy():.4f}, Meta Accuracy: {meta_accuracy.numpy():.4f}")
    
    # Save meta-model
    name_save = model_name + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_maml'
    
    meta_model.save('./network_models/' + name_save + '_meta_model.h5')
    
    # Test MAML
    print("Testing MAML...")
    
    # Select k samples for each class as support set
    support_set_x = []
    support_set_y = []
    
    for class_idx in range(num_classes):
        # Find samples belonging to current class
        class_samples = all_samples_train[all_labels_train == class_idx]
        if len(class_samples) >= k_shot:
            # Select first k samples
            selected_samples = class_samples[:k_shot]
            support_set_x.extend(selected_samples)
            support_set_y.extend([class_idx] * k_shot)
    
    support_set_x = np.array(support_set_x)
    support_set_y = np.array(support_set_y)
    
    # Based on meta-model, create adapted model
    adapted_model = build_maml_model(input_shape, num_classes)
    adapted_model.set_weights(meta_model.get_weights())
    
    # Adapt model on support set (inner loop optimization)
    for adaptation_step in range(5):  # 5 adaptation steps
        with tf.GradientTape() as tape:
            support_logits = adapted_model(support_set_x, training=True)
            support_loss = tf.keras.losses.sparse_categorical_crossentropy(
                support_set_y, support_logits, from_logits=True
            )
            support_loss = tf.reduce_mean(support_loss)
        
        # Calculate gradients
        grads = tape.gradient(support_loss, adapted_model.trainable_variables)
        
        # Update model
        updated_vars = []
        for var, grad in zip(adapted_model.trainable_variables, grads):
            updated_vars.append(var - inner_lr * grad)
        
        adapted_model.set_weights(updated_vars)
    
    # Evaluate adapted model
    total_accuracy = 0
    num_batches = 0
    
    # Prediction results list
    labels_pred_test = []
    
    # Process test samples in batches
    batch_size = 100
    for start_idx in range(0, len(all_samples_test), batch_size):
        end_idx = min(start_idx + batch_size, len(all_samples_test))
        batch_samples = all_samples_test[start_idx:end_idx]
        batch_labels = all_labels_test[start_idx:end_idx]
        
        # Use adapted model for prediction
        logits = adapted_model(batch_samples, training=False)
        predictions = tf.argmax(logits, axis=1).numpy()
        
        # Add to prediction results list
        labels_pred_test.extend(predictions)
        
        # Calculate accuracy
        batch_accuracy = np.mean(predictions == batch_labels)
        total_accuracy += batch_accuracy
        num_batches += 1
    
    # Calculate total accuracy
    accuracy_test = total_accuracy / num_batches if num_batches > 0 else 0
    print(f"Few-shot learning test accuracy: {accuracy_test:.4f}")
    
    # Convert to numpy array
    labels_pred_test = np.array(labels_pred_test)
    
    return labels_pred_test

def evaluate_model(labels_pred_test, labels_complete_test, num_classes, method, n_way, k_shot, 
                  model_net, model_name, tx_antennas_list, rx_antennas_list, pos_train_val, 
                  pos_test, bandwidth):
    """
    Evaluate model performance and save results
    """
    labels_complete_test_array = np.asarray(labels_complete_test)
    # Add these lines before calling confusion_matrix
    print(f"Number of true labels: {len(labels_complete_test_array)}")
    print(f"Number of predicted labels: {len(labels_pred_test)}")

    # Ensure the two arrays have the same length
    if len(labels_complete_test_array) != len(labels_pred_test):
        # Option 1: Truncate to same length
        min_length = min(len(labels_complete_test_array), len(labels_pred_test))
        labels_complete_test_array = labels_complete_test_array[:min_length]
        labels_pred_test = labels_pred_test[:min_length]

    # Calculate evaluation metrics
    conf_matrix_test = confusion_matrix(labels_complete_test_array, labels_pred_test,
                                      labels=list(range(num_classes)),
                                      normalize='true')
    precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(labels_complete_test_array,
                                                                                labels_pred_test,
                                                                                labels=list(range(num_classes)))
    accuracy_test = accuracy_score(labels_complete_test_array, labels_pred_test)
    print('Test accuracy: %.5f' % accuracy_test)
    
    # Save results
    trainable_parameters = np.sum([np.prod(v.get_shape()) for v in model_net.trainable_variables]) if method == 'normal' else 0
    metrics_dict = {'method': method,
                   'n_way': n_way,
                   'k_shot': k_shot,
                   'trainable_parameters': trainable_parameters,
                   'conf_matrix_test': conf_matrix_test, 
                   'accuracy_test': accuracy_test,
                   'precision_test': precision_test, 
                   'recall_test': recall_test, 
                   'fscore_test': fscore_test
                  }
    
    name_save = model_name + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                f'_{method}_n{n_way}_k{k_shot}'
    
    name_file = './outputs/' + name_save + '.txt'
    
    # Create directory (if it doesn't exist)
    directory = os.path.dirname(name_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print("Saving results...")
    with open(name_file, "wb") as fp:
        pickle.dump(metrics_dict, fp)
    
    return metrics_dict