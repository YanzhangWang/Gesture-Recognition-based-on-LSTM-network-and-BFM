import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.layers import Bidirectional, LSTM, BatchNormalization, Reshape, TimeDistributed
from keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling3D, Permute, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras import regularizers

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

def attention_block(x, num_heads=4, key_dim=64):
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention_output + x)
    return x

def optimal_3d_network(input_sh, num_classes):
    # Input layer - shape (None, 234, 4, 2)
    x_input = Input(shape=input_sh)
    
    # Add channel dimension - shape (None, 234, 4, 2, 1)
    x = tf.expand_dims(x_input, axis=-1)
    
    # 3D convolutional layers
    # First convolution block - output shape approx. (None, 117, 4, 2, 64)
    x = Conv3D(64, (7, 2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)
    
    # Second convolution block - output shape approx. (None, 58, 4, 2, 128)
    x = Conv3D(128, (5, 2, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)
    
    # Third convolution block - output shape approx. (None, 29, 2, 2, 256)
    x = Conv3D(256, (3, 2, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)
    
    # Flatten layer for feature extraction
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Reshape features to (None, 2, 256) to introduce sequence structure
    x = Dense(512, activation='relu')(x)
    x = Reshape((2, 256))(x)
    
    # LSTM processing
    x = Bidirectional(LSTM(128))(x)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    logits = Dense(num_classes, activation=None)(x)
    
    # Create model
    model = tf.keras.Model(inputs=x_input, outputs=logits, name='3d_cnn_lstm')
    
    return model


def dynamic_gesture_model(input_shape, num_classes, use_attention=True):
    """
    Build a hybrid CNN-LSTM-Attention model for dynamic gesture recognition
    
    Parameters:
    - input_shape: Input data shape (time_steps, height, width, channels)
    - num_classes: Number of classification categories
    - use_attention: Whether to use self-attention mechanism
    
    Returns:
    - model: Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)

    
    # 3D convolution blocks - extract spatiotemporal features
    x = ConvNormalization3D(32, (3, 3, 3), strides=(1, 1, 1), name_layer='conv3d_1')(inputs)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool_1')(x)
    
    x = ConvNormalization3D(64, (3, 3, 3), strides=(1, 1, 1), name_layer='conv3d_2')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool_2')(x)
    
    x = ConvNormalization3D(128, (3, 3, 3), strides=(1, 1, 1), name_layer='conv3d_3')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool_3')(x)
    
    # Reshape as sequence data while preserving convolutional features
    # Assume the first dimension is the time dimension
    time_steps = input_shape[0]
    
    # Method 1: Use Reshape and TimeDistributed
    # Get the current feature map shape
    conv_features = tf.keras.backend.int_shape(x)
    # Reshape to (batch, time_steps, features)
    x = Reshape((time_steps, -1))(x)
    
    # Bidirectional LSTM to process sequence features
    x = Bidirectional(LSTM(256, return_sequences=True, name='lstm_1'))(x)
    
    # Add self-attention mechanism (optional)
    if use_attention:
        x = attention_block(x)
    
    # Second layer LSTM, not returning sequences
    x = Bidirectional(LSTM(128, return_sequences=False, name='lstm_2'))(x)
    
    # Fully connected classification layers
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Build and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def spatio_temporal_model(input_shape, num_classes):
    time_steps, height, width, channels = input_shape
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Convert input to 5D for 3D convolution
    x = tf.expand_dims(inputs, axis=-1)  # (batch, time_steps, height, width, channels, 1)
    
    # First 3D convolution block - reduce feature dimensions but preserve time dimension
    x = Conv3D(32, kernel_size=(3, 3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(x)  # Downsample only in time dimension
    
    # Second 3D convolution block
    x = Conv3D(64, kernel_size=(3, 3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(x)  # Downsample only in time dimension
    
    # Third 3D convolution block
    x = Conv3D(128, kernel_size=(3, 3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)  # Downsample in time and height dimensions
    
    # Calculate current shape
    current_time_steps = time_steps // 8  # 3 times 2x downsampling in time dimension
    current_height = height // 2  # 1 time 2x downsampling in height
    current_width = width  # Width unchanged
    
    # Spatial dimension compression (preserving time dimension)
    # Reshape tensor to apply 2D convolution to each time step
    x = Reshape((current_time_steps, current_height * current_width * 128))(x)
    
    # Add attention mechanism - helps model focus on important time steps
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=64
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention_output + x)
    
    # Bidirectional LSTM to process time series - using more LSTM units
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)  # Last LSTM doesn't return sequences
    
    # Classification layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name="spatio_temporal_gesture")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Another approach: Using ConvLSTM2D layer
def convlstm_model(input_shape, num_classes):

    time_steps, height, width, channels = input_shape
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Reshape to fit ConvLSTM2D (batch, time, height, width, channels)
    x = inputs
    
    # Spatiotemporal feature extraction
    x = keras.layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Don't reduce time dimension
    
    x = keras.layers.ConvLSTM2D(128, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x = BatchNormalization()(x)
    
    # Global spatial pooling
    x = GlobalAveragePooling2D()(x)
    
    # Classification layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name="convlstm_gesture")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

# Define a better 1D-CNN-LSTM hybrid model
def hybrid_cnn_lstm_model(input_shape, num_classes):

    time_steps, height, width, channels = input_shape
    spatial_dim = height * width * channels
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Reshape to focus on time dimension, flattening spatial dimensions
    # From (batch, time, height, width, channels) to (batch, time, features)
    x = Reshape((time_steps, spatial_dim))(inputs)
    
    # Use 1D convolution to process time dimension
    x = keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = keras.layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Self-attention layer
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=64
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention_output + x)
    
    # LSTM layers
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)
    
    # Classification layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm_gesture")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def residual_static_gesture_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolution block - REDUCED filters from 64 to 32
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual block 1 - REDUCED filters from 64 to 32
    shortcut = x
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # Residual block 2 - REDUCED filters from 128 to 64
    shortcut = Conv2D(64, (1, 1), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    shortcut = BatchNormalization()(shortcut)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # REMOVED Residual block 3 completely
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer - REDUCED neurons from 512 to 256
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.0001))(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs, name="residual_static_gesture_reduced")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Prototypical Network Definition
class PrototypicalNetwork(tf.keras.Model):

    def __init__(self, input_shape):
        super(PrototypicalNetwork, self).__init__()
        
        # Calculate flattened feature size
        self.flat_dim = np.prod(input_shape) * 2  # *2 because complex has real and imaginary parts
        
        # Create feature extractor - with explicit input dimensions
        self.encoder = tf.keras.Sequential([
            # First layer explicitly specifies input dimensions
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
        """Preprocess complex data and extract features"""
        
        # Convert complex data to real features
        if hasattr(x, 'dtype') and 'complex' in str(x.dtype).lower():
            # Extract real and imaginary parts
            real_part = tf.math.real(x)
            imag_part = tf.math.imag(x)
            
            # Flatten and concatenate
            real_flat = tf.reshape(real_part, [tf.shape(x)[0], -1])
            imag_flat = tf.reshape(imag_part, [tf.shape(x)[0], -1])
            features = tf.concat([real_flat, imag_flat], axis=-1)
            
            return tf.cast(features, tf.float32)
        
        elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.complex):
            # Process NumPy complex array
            real_part = np.real(x)
            imag_part = np.imag(x)
            
            # Flatten and concatenate
            real_flat = np.reshape(real_part, (x.shape[0], -1))
            imag_flat = np.reshape(imag_part, (x.shape[0], -1))
            features = np.concatenate([real_flat, imag_flat], axis=-1)
            
            return tf.convert_to_tensor(features, dtype=tf.float32)
        
        else:
            # Process data that is already real
            return tf.reshape(x, [tf.shape(x)[0], -1])
    
    def call(self, support_set, support_labels, query_set, n_way, k_shot, training=False):
        # Process support set
        support_features = self.extract_features(support_set)
        z_support = self.encoder(support_features, training=training)
        
        # Process query set
        query_features = self.extract_features(query_set)
        z_query = self.encoder(query_features, training=training)
        
        # Calculate prototypes
        prototypes = []
        support_labels_np = support_labels
        
        if isinstance(support_labels, tf.Tensor):
            support_labels_np = support_labels.numpy()
        
        for i in range(n_way):
            # Find samples belonging to current class
            class_indices = np.where(support_labels_np == i)[0]
            class_samples = tf.gather(z_support, class_indices)
            
            # Calculate class prototype (mean)
            prototype = tf.reduce_mean(class_samples, axis=0)
            prototypes.append(prototype)
        
        # Stack prototypes into a tensor
        prototypes = tf.stack(prototypes)  # [n_way, feature_dim]
        
        # Calculate Euclidean distance from query samples to each prototype
        dists = []
        for prototype in prototypes:
            # Calculate distance to this prototype
            dist = tf.reduce_sum(tf.square(z_query - tf.expand_dims(prototype, 0)), axis=1)
            dists.append(dist)
        
        dists = tf.stack(dists, axis=1)  # [n_query, n_way]
        
        # Return negative distance as logits
        return -dists
    
    def train_step(self, data):
        # Parse data
        support_x, support_y, query_x, query_y = data
        n_way = tf.reduce_max(support_y).numpy() + 1
        k_shot = tf.shape(support_x)[0] // n_way
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self(support_x, support_y, query_x, n_way, k_shot, training=True)
            
            # Calculate loss - cross entropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                query_y, logits, from_logits=True
            )
            loss = tf.reduce_mean(loss)
        
        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )
        
        return {"loss": loss, "accuracy": accuracy}

def create_siamese_network(input_shape, hidden_dims=[256, 128, 64]):
    # Calculate flattened dimension
    flat_dim = np.prod(input_shape)
    
    # Create feature extractor
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((flat_dim,))  # First flatten the input
    ])
    
    # Add multiple fully connected layers
    for dim in hidden_dims:
        feature_extractor.add(tf.keras.layers.Dense(dim))
        feature_extractor.add(tf.keras.layers.BatchNormalization())
        feature_extractor.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        feature_extractor.add(tf.keras.layers.Dropout(0.3))
    
    # Create dual inputs
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    
    # Shared weight feature extraction
    output_a = feature_extractor(input_a)
    output_b = feature_extractor(input_b)
    
    # Calculate Euclidean distance
    distance = tf.keras.layers.Lambda(
        lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True))
    )([output_a, output_b])
    
    # Create siamese model
    siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance)
    
    return siamese_model, feature_extractor

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def build_maml_model(input_shape, num_classes):
    # Calculate flattened dimension
    flat_dim = np.prod(input_shape)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((flat_dim,)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(num_classes)
    ])
    
    return model

def optimal_2d_sequence_network(input_shape, num_classes, time_steps=10, use_attention=True):
    """
    Build a hybrid 2D CNN-LSTM-Attention model for processing sample sequences, enhancing resistance to overfitting
    
    Parameters:
    - input_shape: Shape of a single sample (num_subcarriers, features, channels) e.g. (234, 4, 2)
    - num_classes: Number of classification categories
    - time_steps: Number of samples in each sequence
    - use_attention: Whether to use self-attention mechanism
    
    Returns:
    - model: Compiled Keras model with input shape (time_steps, num_subcarriers, features, channels)
    """
    # Input layer, shape (time_steps, num_subcarriers, features, channels)
    inputs = tf.keras.layers.Input(shape=(time_steps, *input_shape))
    
    # Define CNN processing block for a single sample, adding regularization and Dropout
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),  # Light Dropout
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # Medium Dropout
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.2),  # Medium Dropout
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        tf.keras.layers.Flatten()
    ])
    
    # Apply the same CNN to each time step (i.e., each sample)
    x = tf.keras.layers.TimeDistributed(cnn_model)(inputs)
    
    # Apply bidirectional LSTM to process sequence features, reduce unit count
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True,
                            recurrent_regularizer=tf.keras.regularizers.l2(0.001))
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Add Dropout after LSTM
    
    # Add self-attention mechanism (if enabled)
    if use_attention:
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention_weights = tf.keras.layers.Flatten()(attention)
        attention_weights = tf.keras.layers.Activation('softmax')(attention_weights)
        attention_weights = tf.keras.layers.RepeatVector(256)(attention_weights)  # 256 = 128*2 (bidirectional)
def optimal_2d_sequence_network(input_shape, num_classes, time_steps=10, use_attention=True):
    """
    Build a hybrid 2D CNN-LSTM-Attention model for processing sample sequences, enhancing resistance to overfitting
    
    Parameters:
    - input_shape: Shape of a single sample (num_subcarriers, features, channels) e.g. (234, 4, 2)
    - num_classes: Number of classification categories
    - time_steps: Number of samples in each sequence
    - use_attention: Whether to use self-attention mechanism
    
    Returns:
    - model: Compiled Keras model with input shape (time_steps, num_subcarriers, features, channels)
    """
    # Input layer, shape (time_steps, num_subcarriers, features, channels)
    inputs = tf.keras.layers.Input(shape=(time_steps, *input_shape))
    
    # Define CNN processing block for a single sample, adding regularization and Dropout
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),  # Light Dropout
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # Medium Dropout
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.2),  # Medium Dropout
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        tf.keras.layers.Flatten()
    ])
    
    # Apply the same CNN to each time step (i.e., each sample)
    x = tf.keras.layers.TimeDistributed(cnn_model)(inputs)
    
    # Apply bidirectional LSTM to process sequence features, reduce unit count
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True,
                            recurrent_regularizer=tf.keras.regularizers.l2(0.001))
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Add Dropout after LSTM
    
    # Add self-attention mechanism (if enabled)
    if use_attention:
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention_weights = tf.keras.layers.Flatten()(attention)
        attention_weights = tf.keras.layers.Activation('softmax')(attention_weights)
        attention_weights = tf.keras.layers.RepeatVector(256)(attention_weights)  # 256 = 128*2 (bidirectional)
        attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
        
        # Apply attention weights
        x = tf.keras.layers.Multiply()([x, attention_weights])
    
    # Second layer LSTM, not returning sequences
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False,
                            recurrent_regularizer=tf.keras.regularizers.l2(0.001))
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # Increase Dropout rate
    
    # Fully connected classification layers
    x = tf.keras.layers.Dense(64, activation='relu',  # Reduce neuron count
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = tf.keras.Model(inputs, outputs)
    
    return model


def optimal_2d_sequence_network2(input_shape, num_classes, time_steps=10, use_attention=True):
    """
    Build a hybrid 2D CNN-LSTM-Attention model for processing sample sequences, enhancing resistance to overfitting
    and improving static gesture recognition capability
    
    Parameters:
    - input_shape: Shape of a single sample (num_subcarriers, features, channels) e.g. (234, 4, 2)
    - num_classes: Number of classification categories
    - time_steps: Number of samples in each sequence
    - use_attention: Whether to use self-attention mechanism
    
    Returns:
    - model: Compiled Keras model with input shape (time_steps, num_subcarriers, features, channels)
    """
    # Input layer, shape (time_steps, num_subcarriers, features, channels)
    inputs = tf.keras.layers.Input(shape=(time_steps, *input_shape))
    
    # Extract dynamic features - calculate differences between adjacent frames
    # This helps distinguish between static and dynamic gestures
    dynamic_features = tf.keras.layers.Lambda(
        lambda x: tf.concat([
            x[:, 0:1, :, :, :],  # Keep first frame
            x[:, 1:, :, :, :] - x[:, :-1, :, :, :]  # Calculate difference
        ], axis=1)
    )(inputs)
    
    # Concatenate original input and dynamic features
    combined_input = tf.keras.layers.Concatenate(axis=-1)([
        inputs, 
        tf.keras.layers.Lambda(lambda x: tf.tile(
            tf.reduce_mean(tf.abs(x), axis=1, keepdims=True), 
            [1, time_steps, 1, 1, 1]
        ))(dynamic_features)
    ])
    
    # Define CNN processing block for a single sample, adding regularization and Dropout
    cnn_model = tf.keras.Sequential([
        # First convolutional layer - extract basic features
        tf.keras.layers.Conv2D(64, (5, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        # Second convolutional layer - extract higher-level features
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        # Third convolutional layer - extract highest-level features
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        # Calculate static features - mean absolute difference
        tf.keras.layers.Lambda(lambda x: tf.reduce_mean(tf.abs(x), axis=[1, 2], keepdims=True)),
        tf.keras.layers.Flatten()
    ])
    
    # Apply the same CNN to each time step
    x = tf.keras.layers.TimeDistributed(cnn_model)(combined_input)
    
    # Calculate time series feature variance - using custom method instead of reduce_variance
    variance_features = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(tf.square(x - tf.reduce_mean(x, axis=1, keepdims=True)), axis=1, keepdims=True)
    )(x)
    variance_features = tf.keras.layers.Flatten()(variance_features)
    
    # Apply bidirectional LSTM to process sequence features
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True,
                            recurrent_regularizer=tf.keras.regularizers.l2(0.001),
                            recurrent_dropout=0.1)  # Add recurrent layer dropout
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Add self-attention mechanism (if enabled)
    if use_attention:
        # MultiHeadAttention requires TensorFlow 2.4+, if version is lower, may need alternative
        try:
            # Multi-head self-attention mechanism - better capture different types of patterns
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=32
            )(x, x)
            x = tf.keras.layers.Add()([x, attention])  # Residual connection
            x = tf.keras.layers.LayerNormalization()(x)  # Layer normalization
        except:
            # If MultiHeadAttention is unavailable, use custom attention mechanism
            attention = tf.keras.layers.Dense(1, activation='tanh')(x)
            attention_weights = tf.keras.layers.Flatten()(attention)
            attention_weights = tf.keras.layers.Activation('softmax')(attention_weights)
            attention_weights = tf.keras.layers.RepeatVector(256)(attention_weights)  # 256 = 128*2 (bidirectional)
            attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
            # Apply attention weights
            x = tf.keras.layers.Multiply()([x, attention_weights])
    
    # Global temporal feature extraction
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Concatenate all features
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool, variance_features])
    
    # Fully connected classification layers
    x = tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = tf.keras.Model(inputs, outputs)
    return model