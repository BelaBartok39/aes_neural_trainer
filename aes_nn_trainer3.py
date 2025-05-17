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
BATCH_SIZE = 512  # Increased from 128 to better utilize GPU
SEED = 42
MIXED_PRECISION = True
LEARNING_RATE = 2e-3  # Slightly increased learning rate
MIN_LEARNING_RATE = 1e-6
MODEL_CHECKPOINT_PATH = "aes_nn_models"
USE_ATTENTION = True  # Added attention mechanisms
MODEL_TYPE = "transformer"  # Options: "dense", "cnn", "transformer", "hybrid"
XLA_ACCELERATION = True  # Enable XLA compilation for faster execution
MAX_BITS_TO_PREDICT = 32  # Try to predict only first 32 bits instead of all 128

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
    
    # Reshape inputs to add sequence dimension for attention
    # (batch_size, 256) -> (batch_size, 16, 16)
    x = layers.Reshape((16, 16))(inputs)
    
    # Positional encoding
    pos_encoding = positional_encoding(16, 16)
    x = x + pos_encoding
    
    # Transformer encoder blocks
    for _ in range(6):  # 6 transformer blocks
        x = attention_block(x, num_heads=4)
    
    # Global attention pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_cnn_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create a CNN-based model for AES prediction"""
    inputs = layers.Input(shape=input_shape)
    
    # Reshape for 1D convolutions: (batch, 256) -> (batch, 256, 1)
    x = layers.Reshape((256, 1))(inputs)
    
    # First conv block
    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Second conv block
    x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Third conv block
    x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_hybrid_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create a hybrid model combining CNN, LSTM and attention"""
    inputs = layers.Input(shape=input_shape)
    
    # Reshape for sequence processing: (batch, 256) -> (batch, 32, 8)
    x = layers.Reshape((32, 8))(inputs)
    
    # CNN layers for feature extraction
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Self-attention layer
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Bidirectional LSTM for sequence modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Residual block for Dense model
def residual_block(x, units, dropout_rate=0.3, l2_reg=1e-5):
    """Residual block for the dense model"""
    skip = x
    
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    if skip.shape[-1] != units:
        skip = layers.Dense(units, kernel_initializer='he_normal')(skip)
    
    x = layers.Add()([x, skip])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    return x

def create_dense_model(input_shape=(256,), output_shape=MAX_BITS_TO_PREDICT):
    """Create an enhanced dense model with residual connections"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial expansion
    x = layers.Dense(1024, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # First stack of residual blocks
    for i in range(4):  # Increased from 3 to 4
        x = residual_block(x, 1024, dropout_rate=0.3, l2_reg=1e-5)
    
    # Self-attention if enabled
    if USE_ATTENTION:
        # Reshape for attention: (batch, 1024) -> (batch, 32, 32)
        attention_reshape = layers.Reshape((32, 32))(x)
        attn_output = layers.MultiHeadAttention(num_heads=8, key_dim=4)(
            attention_reshape, attention_reshape)
        attn_output = layers.Reshape((1024,))(attn_output)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Transition
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Second stack of residual blocks
    for i in range(3):  # Increased from 2 to 3
        x = residual_block(x, 512, dropout_rate=0.3, l2_reg=1e-5)
    
    # Output layer
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
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 16
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=min(100000, X_train.shape[0]), seed=SEED)
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
    
    print("tf.data.Dataset pipelines optimized with performance options")

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
        ]
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
            patience=15,
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
            update_freq='epoch'
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
        callbacks.CSVLogger(f'training_log_{MODEL_TYPE}.csv', append=True)
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

