import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, regularizers
from Crypto.Cipher import AES
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# --- Enhanced Configuration with GPU Optimization ---
NUM_SAMPLES = 1000000  # 1M samples
EPOCHS = 100
BATCH_SIZE = 512  # Increased to 512 for better GPU utilization
SEED = 42
MIXED_PRECISION = True
LEARNING_RATE = 2e-3  # Slightly increased learning rate
MIN_LEARNING_RATE = 1e-6
MODEL_CHECKPOINT_PATH = "aes_nn_models"
USE_ATTENTION = True  # Added attention mechanisms
MODEL_TYPE = "hybrid"  # Options: "dense", "cnn", "transformer", "hybrid"
XLA_ACCELERATION = True  # Enable XLA compilation for faster execution
MAX_BITS_TO_PREDICT = 32  # Try to predict only first 32 bits instead of all 128
MODEL_COMPLEXITY = "ultra"  # Options: "normal", "high", "ultra"

# Create checkpoint directory if it doesn't exist
os.makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable XLA compilation for faster execution
if XLA_ACCELERATION:
    tf.config.optimizer.set_jit(True)
    print("XLA compilation enabled")

# Enable mixed precision
if MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled with policy:", policy)

# For tf.data API
AUTOTUNE = tf.data.AUTOTUNE

# --- Data Generation ---
def generate_sample():
    """Generate a single encryption sample"""
    key = np.random.bytes(16)
    plaintext = np.random.bytes(16)
    
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(plaintext)
    
    pt_bits = np.unpackbits(np.frombuffer(plaintext, dtype=np.uint8))
    key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
    ct_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
    
    # If we're only predicting a subset of bits
    if MAX_BITS_TO_PREDICT < 128:
        ct_bits = ct_bits[:MAX_BITS_TO_PREDICT]
    
    return pt_bits, key_bits, ct_bits

# Worker function (at module level for multiprocessing)
def worker_task(args):
    """Generate samples for a specific range"""
    worker_id, start_idx, end_idx = args
    
    worker_X = np.zeros((end_idx - start_idx, 256), dtype=np.float32)
    worker_y = np.zeros((end_idx - start_idx, MAX_BITS_TO_PREDICT), dtype=np.float32)
    
    for i in range(end_idx - start_idx):
        pt_bits, key_bits, ct_bits = generate_sample()
        worker_X[i] = np.concatenate([pt_bits, key_bits])
        worker_y[i] = ct_bits
        
    return worker_id, worker_X, worker_y

def generate_dataset_parallel(num_samples, num_workers=8):
    """Generate dataset using parallel processing"""
    from concurrent.futures import ProcessPoolExecutor
    
    X_data = np.zeros((num_samples, 256), dtype=np.float32)
    y_data = np.zeros((num_samples, MAX_BITS_TO_PREDICT), dtype=np.float32)
    
    print(f"Generating {num_samples} samples using {num_workers} workers...")
    start_time = time.time()
    
    # Divide work among workers
    samples_per_worker = num_samples // num_workers
    
    # Prepare arguments for each worker
    worker_args = []
    for i in range(num_workers):
        start_idx = i * samples_per_worker
        end_idx = start_idx + samples_per_worker if i < num_workers - 1 else num_samples
        worker_args.append((i, start_idx, end_idx))
    
    try:
        # Execute workers in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, args) for args in worker_args]
            
            # Use tqdm for a progress bar
            for future in tqdm(futures, total=num_workers):
                worker_id, worker_X, worker_y = future.result()
                start_idx = worker_id * samples_per_worker
                end_idx = start_idx + samples_per_worker if worker_id < num_workers - 1 else num_samples
                X_data[start_idx:end_idx] = worker_X
                y_data[start_idx:end_idx] = worker_y
    except Exception as e:
        print(f"Error with parallel processing: {e}")
        print("Falling back to single-process data generation...")
        return generate_dataset_sequential(num_samples)
    
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds.")
    return X_data, y_data

def generate_dataset_sequential(num_samples):
    """Sequential data generation fallback"""
    print(f"Generating {num_samples} samples sequentially...")
    start_time = time.time()
    
    X_data = np.zeros((num_samples, 256), dtype=np.float32)
    y_data = np.zeros((num_samples, MAX_BITS_TO_PREDICT), dtype=np.float32)
    
    # Use a smaller number of samples for the sequential method to save time
    actual_samples = min(num_samples, 500000)
    if actual_samples < num_samples:
        print(f"Warning: Reducing to {actual_samples} samples for sequential processing")
    
    for i in tqdm(range(actual_samples)):
        pt_bits, key_bits, ct_bits = generate_sample()
        X_data[i] = np.concatenate([pt_bits, key_bits])
        y_data[i] = ct_bits
    
    end_time = time.time()
    print(f"Sequential dataset generation took {end_time - start_time:.2f} seconds.")
    return X_data[:actual_samples], y_data[:actual_samples]

# --- Enhanced Model Architectures ---

# Self-Attention mechanism
def attention_block(x, num_heads=4):
    """Multi-head self-attention block"""
    dim = x.shape[-1]
    
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=dim // num_heads
    )(x, x)
    
    # Add & Normalize
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn = layers.Dense(dim * 2, activation='relu')(x)
    ffn = layers.Dense(dim)(ffn)
    
    # Add & Normalize
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x

def create_transformer_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create a Transformer-based model for AES prediction"""
    inputs = layers.Input(shape=input_shape)
    
    # Choose layer configuration based on complexity setting
    if MODEL_COMPLEXITY == "normal":
        embed_dim = 256
        num_heads = 4
        num_blocks = 6
        ff_dim = 512
    elif MODEL_COMPLEXITY == "high":
        embed_dim = 512
        num_heads = 8
        num_blocks = 8
        ff_dim = 1024
    elif MODEL_COMPLEXITY == "ultra":
        embed_dim = 768
        num_heads = 12
        num_blocks = 12
        ff_dim = 2048
    
    # Initial dense layer to project to embedding dimension
    x = layers.Dense(embed_dim)(inputs)
    
    # Reshape inputs to add sequence dimension for attention
    # Split into 16 segments of embed_dim/16 dimensions each
    seq_length = 16
    feature_dim = embed_dim // seq_length
    x = layers.Reshape((seq_length, feature_dim))(x)
    
    # Positional encoding
    pos_encoding = positional_encoding(seq_length, feature_dim)
    x = x + pos_encoding
    
    # Transformer encoder blocks
    for _ in range(num_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=feature_dim // num_heads
        )(x, x)
        
        # Add & Normalize (first residual connection)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation='gelu')(x)  # Using GELU instead of ReLU
        ffn = layers.Dense(feature_dim)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        
        # Add & Normalize (second residual connection)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Two different pooling approaches
    # 1. Global average pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    
    # 2. Flatten
    flat = layers.Flatten()(x)
    
    # Combine both approaches
    x = layers.Concatenate()([avg_pool, flat])
    
    # Final dense layers - multiple stacked layers with dropout
    x = layers.Dense(1024, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_cnn_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create a CNN-based model for AES prediction"""
    inputs = layers.Input(shape=input_shape)
    
    # Choose layer configuration based on complexity setting
    if MODEL_COMPLEXITY == "normal":
        filters = [256, 256, 512]
        kernel_sizes = [3, 3, 3]
        dense_units = [512, 256]
    elif MODEL_COMPLEXITY == "high":
        filters = [256, 384, 512, 768]
        kernel_sizes = [3, 5, 3, 3]
        dense_units = [768, 384]
    elif MODEL_COMPLEXITY == "ultra":
        filters = [256, 384, 512, 768, 1024]
        kernel_sizes = [3, 5, 7, 3, 3]
        dense_units = [1024, 512, 256]
    
    # Multiple input representations for different feature extraction
    
    # Representation 1: 1D sequence
    x1 = layers.Reshape((256, 1))(inputs)
    
    # First conv block - process full sequence
    for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
        x1 = layers.Conv1D(f, kernel_size=k, padding='same', activation=None)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('gelu')(x1)  # Using GELU
        
        # Add residual connections where possible
        if i > 0 and filters[i] == filters[i-1]:
            res_conn = layers.Conv1D(f, kernel_size=1, padding='same')(x1)
            x1 = layers.Add()([x1, res_conn])
        
        # Downsampling every other layer
        if i % 2 == 1 and i < len(filters) - 1:
            x1 = layers.MaxPooling1D(pool_size=2)(x1)
    
    # Global pooling
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # Representation 2: 2D matrix for 2D convolutions - capture 2D patterns
    x2 = layers.Reshape((16, 16, 1))(inputs)
    
    # 2D Convolutional path
    x2 = layers.Conv2D(64, kernel_size=3, padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('gelu')(x2)
    x2 = layers.MaxPooling2D(pool_size=2)(x2)
    
    x2 = layers.Conv2D(128, kernel_size=3, padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('gelu')(x2)
    x2 = layers.MaxPooling2D(pool_size=2)(x2)
    
    x2 = layers.Conv2D(256, kernel_size=3, padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('gelu')(x2)
    
    # Global pooling
    x2 = layers.GlobalAveragePooling2D()(x2)
    
    # Combine representations
    x = layers.Concatenate()([x1, x2])
    
    # Dense layers with residual connections
    for units in dense_units:
        res = x
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Add residual connection if shapes allow
        if res.shape[-1] == units:
            x = layers.Add()([x, res])
        elif hasattr(res, 'shape'):
            # Project residual to match shape
            res = layers.Dense(units, kernel_initializer='he_normal')(res)
            x = layers.Add()([x, res])
    
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_hybrid_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create a hybrid model combining CNN, LSTM and attention"""
    inputs = layers.Input(shape=input_shape)
    
    # Choose layer configuration based on complexity setting
    if MODEL_COMPLEXITY == "normal":
        lstm_units = 128
        cnn_filters = [128, 128, 128]
        dense_units = [512, 256]
        attention_heads = 4
    elif MODEL_COMPLEXITY == "high":
        lstm_units = 256
        cnn_filters = [128, 256, 256]
        dense_units = [768, 384]
        attention_heads = 8
    elif MODEL_COMPLEXITY == "ultra":
        lstm_units = 384
        cnn_filters = [256, 384, 512]
        dense_units = [1024, 512, 256]
        attention_heads = 12
    
    # Multiple pathways for different representations
    
    # Path 1: Reshape for CNN-1D processing: (batch, 256) -> (batch, 256, 1)
    cnn_input = layers.Reshape((256, 1))(inputs)
    
    # CNN pathway with residual connections
    x_cnn = cnn_input
    for i, filters in enumerate(cnn_filters):
        # CNN block
        conv = layers.Conv1D(filters, kernel_size=3, padding='same')(x_cnn)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('gelu')(conv)
        
        # Residual connection if shapes match
        if i > 0 and cnn_filters[i] == cnn_filters[i-1]:
            x_cnn = layers.Add()([x_cnn, conv])
        else:
            x_cnn = conv
        
        # Add pooling every other layer
        if i % 2 == 1:
            x_cnn = layers.MaxPooling1D(pool_size=2)(x_cnn)
    
    # Path 2: Reshape for sequence processing: (batch, 256) -> (batch, 32, 8)
    seq_input = layers.Reshape((32, 8))(inputs)
    
    # Add positional encoding for sequence
    pos_encoding = positional_encoding(32, 8)
    x_seq = seq_input + pos_encoding
    
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=attention_heads, 
        key_dim=8 // min(8, attention_heads)
    )(x_seq, x_seq)
    
    # Residual with self-attention
    x_seq = layers.Add()([x_seq, attn_output])
    x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq)
    
    # Bidirectional LSTM for sequence modeling - stacked
    x_seq = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x_seq)
    x_seq = layers.Dropout(0.3)(x_seq)
    x_seq = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True))(x_seq)
    x_seq = layers.Dropout(0.3)(x_seq)
    
    # Global pooling for each pathway
    x_cnn_pool = layers.GlobalAveragePooling1D()(x_cnn)
    x_seq_pool = layers.GlobalAveragePooling1D()(x_seq)
    
    # Combine pathways
    x = layers.Concatenate()([x_cnn_pool, x_seq_pool])
    
    # Dense layers
    for units in dense_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Residual block for Dense model
def residual_block(x, units, dropout_rate=0.3, l2_reg=1e-5, activation='relu'):
    """Residual block for the dense model"""
    skip = x
    
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    if activation == 'relu':
        x = layers.Activation('relu')(x)
    elif activation == 'gelu':
        x = tf.keras.activations.gelu(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    if skip.shape[-1] != units:
        skip = layers.Dense(units, kernel_initializer='he_normal')(skip)
    
    x = layers.Add()([x, skip])
    
    if activation == 'relu':
        x = layers.Activation('relu')(x)
    elif activation == 'gelu':
        x = tf.keras.activations.gelu(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    return x

def create_dense_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create an enhanced dense model with residual connections"""
    inputs = layers.Input(shape=input_shape)
    
    # Choose layer configuration based on complexity setting
    if MODEL_COMPLEXITY == "normal":
        units = [1024, 1024, 512]
        blocks_per_size = [3, 2]
    elif MODEL_COMPLEXITY == "high":
        units = [1536, 1536, 768, 768]
        blocks_per_size = [4, 3]
    elif MODEL_COMPLEXITY == "ultra":
        units = [2048, 2048, 1024, 1024, 512]
        blocks_per_size = [5, 4, 3]
    
    # Initial expansion
    x = layers.Dense(units[0], kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)  # Using GELU instead of ReLU
    
    # First stack of residual blocks
    for i in range(blocks_per_size[0]):
        x = residual_block(x, units[0], dropout_rate=0.3, l2_reg=1e-5, activation='gelu')
    
    # Self-attention if enabled
    if USE_ATTENTION:
        # Choose attention configuration based on complexity
        if MODEL_COMPLEXITY == "normal":
            attention_heads = 8
            attention_dim = 32
        elif MODEL_COMPLEXITY == "high":
            attention_heads = 12
            attention_dim = 48
        else:  # ultra
            attention_heads = 16
            attention_dim = 64
            
        # Reshape for attention: (batch, units[0]) -> (batch, reshape_dim, feature_dim)
        reshape_dim = attention_dim
        feature_dim = units[0] // reshape_dim
        attention_reshape = layers.Reshape((reshape_dim, feature_dim))(x)
        
        attn_output = layers.MultiHeadAttention(
            num_heads=attention_heads, 
            key_dim=feature_dim // min(feature_dim, attention_heads)
        )(attention_reshape, attention_reshape)
        
        attn_output = layers.Reshape((units[0],))(attn_output)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Process through remaining unit sizes
    for i in range(1, len(units)):
        # Transition to next size
        x = layers.Dense(units[i], kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual blocks at this size
        blocks_at_this_size = blocks_per_size[min(i, len(blocks_per_size)-1)]
        for j in range(blocks_at_this_size):
            x = residual_block(x, units[i], dropout_rate=0.3, l2_reg=1e-5, activation='gelu')
    
    # Final output layer
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Position encoding for transformer
def positional_encoding(length, depth):
    """Create positional encodings for the transformer"""
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / (10000**(depths / depth))
    angle_rads = positions * angle_rates
    
    pos_encoding = np.zeros((1, length, depth))
    pos_encoding[0, :, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[0, :, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_model(model_type=MODEL_TYPE, input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create model based on selected architecture type"""
    print(f"Creating {model_type.upper()} model to predict {output_shape} bits...")
    
    if model_type == "dense":
        return create_dense_model(input_shape, output_shape)
    elif model_type == "cnn":
        return create_cnn_model(input_shape, output_shape)
    elif model_type == "transformer":
        return create_transformer_model(input_shape, output_shape)
    elif model_type == "hybrid":
        return create_hybrid_model(input_shape, output_shape)
    else:
        print(f"Unknown model type: {model_type}, falling back to dense model")
        return create_dense_model(input_shape, output_shape)

# --- Utility Functions ---
def plot_training_history(history, filename="training_history.png"):
    """Plot training metrics over time"""
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    available_metrics = [m for m in metrics if m in history.history]
    
    n_metrics = len(available_metrics)
    fig_cols = min(3, n_metrics)
    fig_rows = (n_metrics + fig_cols - 1) // fig_cols
    
    plt.figure(figsize=(6*fig_cols, 4*fig_rows))
    
    for i, metric in enumerate(available_metrics):
        plt.subplot(fig_rows, fig_cols, i+1)
        plt.plot(history.history[metric], label=f'Training {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric}')
        plt.title(f'Model {metric}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Training history plot saved as {filename}")

def estimate_training_time(model, train_dataset, batch_size, epochs):
    """Estimate training time based on a few test iterations"""
    print("Estimating training time...")
    # Get sample batch for warmup
    for batch in train_dataset.take(1):
        sample_batch = batch
        break
    
    # Warmup pass
    model.predict(sample_batch[0], verbose=0)
    
    # Time a few batches
    n_test_batches = 10
    start_time = time.time()
    
    for _ in range(n_test_batches):
        model.predict(sample_batch[0], verbose=0)
    
    end_time = time.time()
    time_per_batch = (end_time - start_time) / n_test_batches
    
    # Calculate total batches
    steps_per_epoch = len(list(train_dataset))
    total_batches = steps_per_epoch * epochs
    
    # Estimate total time
    estimated_time = total_batches * time_per_batch
    
    # Convert to hours, minutes, seconds
    hours = int(estimated_time // 3600)
    minutes = int((estimated_time % 3600) // 60)
    seconds = int(estimated_time % 60)
    
    print(f"Estimated training time: {hours}h {minutes}m {seconds}s")
    print(f"(Based on {time_per_batch:.4f} seconds per batch)")
    
    return estimated_time

def analyze_bit_level_accuracy(model, test_dataset, y_test, n_samples=1000):
    """Analyze accuracy at the individual bit level"""
    print("\n--- Bit-Level Analysis ---")
    
    # Get predictions for a subset of test data
    predictions = model.predict(test_dataset.take(n_samples // BATCH_SIZE + 1))
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Ensure y_test is sized correctly
    y_subset = y_test[:len(binary_predictions)]
    
    # Calculate overall bit-level accuracy
    bit_accuracy = np.mean(binary_predictions == y_subset.reshape(binary_predictions.shape))
    print(f"Overall bit-level accuracy: {bit_accuracy:.4f} ({bit_accuracy*100:.2f}%)")
    
    # Calculate accuracy per bit position
    bit_accuracies = np.mean(binary_predictions == y_subset.reshape(binary_predictions.shape), axis=0)
    
    # Find best and worst performing bit positions
    best_bit = np.argmax(bit_accuracies)
    worst_bit = np.argmin(bit_accuracies)
    
    print(f"Best bit position: {best_bit} with accuracy {bit_accuracies[best_bit]:.4f} ({bit_accuracies[best_bit]*100:.2f}%)")
    print(f"Worst bit position: {worst_bit} with accuracy {bit_accuracies[worst_bit]:.4f} ({bit_accuracies[worst_bit]*100:.2f}%)")
    
    # Count bits performing above random chance (0.51)
    better_than_random = np.sum(bit_accuracies > 0.51)
    print(f"Bits performing better than random chance (>51%): {better_than_random}/{len(bit_accuracies)} ({better_than_random/len(bit_accuracies)*100:.2f}%)")
    
    # Plot bit accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(bit_accuracies)), bit_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.axhline(y=0.51, color='g', linestyle='--', label='Slightly better than random')
    plt.xlabel('Bit Position')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Bit Position')
    plt.legend()
    plt.savefig("bit_position_accuracy.png", dpi=300)
    
    return bit_accuracies

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(f"Starting Enhanced AES Neural Network Training with {MODEL_TYPE.upper()} architecture")
    print(f"Predicting first {MAX_BITS_TO_PREDICT} bits of ciphertext")
    print("="*50)
    
    # Check for GPU and optimize
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth and optimization
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). TensorFlow will use GPU: {gpus}")
            
            # Print GPU details
            physical_devices = tf.config.list_physical_devices('GPU')
            details = tf.config.experimental.get_device_details(physical_devices[0])
            print(f"GPU details: {details}")
            
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found. TensorFlow will use CPU.")

    # 1. Generate Data
    print("\n" + "-"*50)
    print("Data Generation")
    print("-"*50)
    
    try:
        X, y = generate_dataset_parallel(NUM_SAMPLES)
        print(f"Generated dataset with X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"Error with multiprocessing: {e}")
        print("Using reduced dataset size with sequential processing...")
        reduced_samples = 500000
        X, y = generate_dataset_sequential(reduced_samples)
        print(f"Generated dataset with X shape: {X.shape}, y shape: {y.shape}")

    # 2. Split Dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Release memory
    del X, y, X_temp, y_temp
    
    # 3. Prepare TensorFlow Datasets with performance optimizations
    print("\n" + "-"*50)
    print("Preparing TensorFlow Datasets with optimizations")
    print("-"*50)
    
    # More aggressive performance tuning for tf.data pipeline
    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 48  # Increased from 16 to 48
    
    # Apply optimization options that are available in your TensorFlow version
    # Use try-except to handle API differences between TensorFlow versions
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        # Additional optimizations - wrapped in try-except
        try:
            options.experimental_optimization.map_and_batch_fusion = True
        except AttributeError:
            pass
        try:
            options.experimental_optimization.noop_elimination = True
        except AttributeError:
            pass
        try:
            options.experimental_optimization.shuffle_and_repeat_fusion = True
        except AttributeError:
            pass
    except AttributeError:
        # Fallback for newer TensorFlow versions that moved options
        try:
            options.optimization.map_parallelization = True
            options.optimization.parallel_batch = True
            options.optimization.map_and_batch_fusion = True
            options.optimization.noop_elimination = True
            options.optimization.shuffle_and_repeat_fusion = True
        except AttributeError:
            print("Warning: Some optimization options are not available in your TensorFlow version")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=min(200000, X_train.shape[0]), seed=SEED)  # Larger shuffle buffer
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    train_dataset = train_dataset.with_options(options)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.with_options(options)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.with_options(options)
    
    print("tf.data.Dataset pipelines optimized with available performance options")

    # 4. Create and Compile Model
    print("\n" + "-"*50)
    print(f"Creating {MODEL_TYPE.upper()} Model")
    print("-"*50)
    
    model = create_model(model_type=MODEL_TYPE, input_shape=(X_train.shape[1],), output_shape=MAX_BITS_TO_PREDICT)
    
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5,
        epsilon=1e-7
    )
    
    # Ensure optimizer uses correct precision
    if MIXED_PRECISION:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Compile model with binary focal loss for better handling of difficult bits
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ],
        # Enable XLA compilation for the model
        jit_compile=XLA_ACCELERATION
    )
    
    model.summary()
    
    # 5. Estimate Training Time
    estimated_time = estimate_training_time(model, train_dataset, BATCH_SIZE, EPOCHS)
    
    # 6. Define Callbacks
    print("\n" + "-"*50)
    print("Setting up Advanced Training Callbacks")
    print("-"*50)
    
    # Learning rate scheduler with warmup and cosine decay
    def lr_schedule(epoch):
        # Warmup for the first 5 epochs
        if epoch < 5:
            return LEARNING_RATE * ((epoch + 1) / 5)
        # Cosine decay for the rest
        else:
            decay_epochs = EPOCHS - 5
            epoch_normalized = (epoch - 5) / decay_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_normalized))
            return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine_decay
    
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased from 15 to 20
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_CHECKPOINT_PATH, f'aes_{MODEL_TYPE}_model_e{{epoch:02d}}_vl{{val_loss:.4f}}.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch',
            profile_batch='100,120'  # Enable profiling for batches 100-120
        ),
        
        # Reduce LR on plateau as backup
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # CSV logger
        callbacks.CSVLogger(f'training_log_{MODEL_TYPE}_{MODEL_COMPLEXITY}.csv', append=True)
    ]

    # 7. Train the Model
    print("\n" + "-"*50)
    print(f"Starting {MODEL_TYPE.upper()} Model Training")
    print("-"*50)
    
    # Additional info before training starts
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Predicting {MAX_BITS_TO_PREDICT} bits of the ciphertext")
    if MAX_BITS_TO_PREDICT < 128:
        print("Note: Predicting only a subset of bits to make the learning problem easier")
    
    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        verbose=1
    )
    training_time = time.time() - start_time
    
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"Training completed in {hours}h {minutes}m {seconds}s")
    
    # Compare with our estimate
    print(f"Estimated time was: {int(estimated_time//3600)}h {int((estimated_time%3600)//60)}m {int(estimated_time%60)}s")
    error_percent = abs(training_time - estimated_time) / estimated_time * 100
    print(f"Estimation error: {error_percent:.2f}%")

    # 8. Plot Training History
    plot_training_history(history, filename=f"training_history_{MODEL_TYPE}.png")

    # 9. Load Best Model and Evaluate
    print("\n" + "-"*50)
    print("Loading Best Model and Evaluating")
    print("-"*50)
    
    best_model_path = os.path.join(MODEL_CHECKPOINT_PATH, sorted([f for f in os.listdir(MODEL_CHECKPOINT_PATH) 
                                                                 if f.endswith('.keras') and MODEL_TYPE in f])[-1])
    print(f"Loading best model from: {best_model_path}")
    best_model = keras.models.load_model(best_model_path)
    
    # Evaluate on test set
    metrics = best_model.evaluate(test_dataset, verbose=1)
    metrics_names = best_model.metrics_names
    
    print("\nTest Metrics:")
    for name, value in zip(metrics_names, metrics):
        print(f"  {name}: {value:.6f}")
    
    # 10. Perform bit-level analysis (key for understanding cryptographic learning)
    bit_accuracies = analyze_bit_level_accuracy(best_model, test_dataset, y_test)

    # 11. Save Final Model and Results
    print("\n" + "-"*50)
    print("Saving Final Model and Results")
    print("-"*50)
    
    final_model_path = f"aes_nn_{MODEL_TYPE}_final_model.keras"
    best_model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save bit accuracies to CSV
    np.savetxt(f"bit_accuracies_{MODEL_TYPE}.csv", bit_accuracies, delimiter=",", header="accuracy", comments="")
    print(f"Bit accuracies saved to: bit_accuracies_{MODEL_TYPE}.csv")

    print("\n" + "="*50)
    print("AES Neural Network Training Complete")
    print("="*50)
    print("\nNotes:")
    print("1. Learning AES is an extremely challenging task by design")
    print("2. Any bit position consistently above 51% accuracy would be cryptographically significant")
    print("3. Consider experimenting with different model architectures by changing MODEL_TYPE")
    print("4. To improve results, try targeting an even smaller subset of bits (e.g., just the first 8)")

