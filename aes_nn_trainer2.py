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

# --- Enhanced Configuration ---
NUM_SAMPLES = 1000000  # Reduced from 2M to be more practical while still large
EPOCHS = 100  # More epochs for deeper learning
BATCH_SIZE = 128  # Optimized for RTX 3070
SEED = 42  # Random seed for reproducibility
MIXED_PRECISION = True  # Enable mixed precision for faster computation
LEARNING_RATE = 1e-3  # Initial learning rate
MIN_LEARNING_RATE = 1e-6  # Minimum learning rate for decay
MODEL_CHECKPOINT_PATH = "aes_nn_models"  # Directory to save model checkpoints
USE_RESIDUAL_CONNECTIONS = True  # Enable residual connections

# Create checkpoint directory if it doesn't exist
os.makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable mixed precision for faster computation on RTX 3070
if MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled with policy:", policy)

# For tf.data API
AUTOTUNE = tf.data.AUTOTUNE

# --- 1. Enhanced Data Generation ---
def generate_sample():
    """
    Generates a single sample of (plaintext, key, ciphertext).
    Plaintext and key are 128 bits each. Ciphertext is 128 bits.
    All are returned as binary vectors (numpy arrays of 0s and 1s).
    """
    key = np.random.bytes(16)  # 128-bit key
    plaintext = np.random.bytes(16)  # 128-bit plaintext
    
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(plaintext)
    
    # Convert byte strings to binary vectors (numpy arrays of 0s and 1s)
    pt_bits = np.unpackbits(np.frombuffer(plaintext, dtype=np.uint8))
    key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
    ct_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
    
    return pt_bits, key_bits, ct_bits

# Worker function must be at module level for multiprocessing
def worker_task(args):
    """Generate samples for a specific range"""
    worker_id, start_idx, end_idx = args
    
    worker_X = np.zeros((end_idx - start_idx, 256), dtype=np.float32)
    worker_y = np.zeros((end_idx - start_idx, 128), dtype=np.float32)
    
    for i in range(end_idx - start_idx):
        pt_bits, key_bits, ct_bits = generate_sample()
        worker_X[i] = np.concatenate([pt_bits, key_bits])
        worker_y[i] = ct_bits
        
    return worker_id, worker_X, worker_y

def generate_dataset_parallel(num_samples, num_workers=8):
    """
    Generates a dataset of (plaintext, key, ciphertext) tuples using parallel processing.
    X consists of concatenated [plaintext_bits, key_bits].
    y consists of ciphertext_bits.
    """
    from concurrent.futures import ProcessPoolExecutor
    
    X_data = np.zeros((num_samples, 256), dtype=np.float32)
    y_data = np.zeros((num_samples, 128), dtype=np.float32)
    
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
    """
    Fallback function that generates data sequentially if parallel processing fails.
    """
    print(f"Generating {num_samples} samples sequentially...")
    start_time = time.time()
    
    X_data = np.zeros((num_samples, 256), dtype=np.float32)
    y_data = np.zeros((num_samples, 128), dtype=np.float32)
    
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

# --- 2. Enhanced Model Architecture ---
def residual_block(x, units, dropout_rate=0.3, l2_reg=1e-5):
    """Creates a residual block with skip connection"""
    # Store input for skip connection
    skip = x
    
    # First dense layer with batch normalization and activation
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second dense layer with batch normalization
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection (with projection if needed)
    if skip.shape[-1] != units:
        skip = layers.Dense(units, kernel_initializer='he_normal')(skip)
    
    # Add skip connection
    x = layers.Add()([x, skip])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    return x

def create_enhanced_model(input_shape=(256,), output_shape=128):
    """
    Defines and compiles an enhanced neural network model with residual connections.
    Input: 256 units (128 for plaintext + 128 for key)
    Hidden layers: Multiple residual blocks with skip connections
    Output layer: 128 units with sigmoid (for bitwise binary output)
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # Initial layer to expand dimensionality
    x = layers.Dense(1024, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks
    if USE_RESIDUAL_CONNECTIONS:
        # First stack of residual blocks
        for i in range(3):
            x = residual_block(x, 1024, dropout_rate=0.3, l2_reg=1e-5)
        
        # Transition with dimension reduction
        x = layers.Dense(512, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Second stack of residual blocks
        for i in range(2):
            x = residual_block(x, 512, dropout_rate=0.3, l2_reg=1e-5)
    else:
        # Regular dense layers if residual connections are disabled
        for i in range(5):
            x = layers.Dense(1024, kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(512, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(output_shape, activation='sigmoid', name="output_layer")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Ensure optimizer uses correct precision
    if MIXED_PRECISION:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# --- Improved Utility Functions ---
def plot_training_history(history, filename="training_history.png"):
    """Plots training and validation metrics."""
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

def analyze_predictions(model, test_dataset, y_test, n_samples=1000):
    """Performs detailed analysis of the model's predictions."""
    print("\n--- Detailed Prediction Analysis ---")
    
    # Get predictions for a subset of test data for detailed analysis
    predictions = model.predict(test_dataset.take(n_samples // BATCH_SIZE + 1))
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate overall bitwise accuracy
    correct_bits = np.sum(binary_predictions == y_test[:len(binary_predictions)].reshape(binary_predictions.shape))
    total_bits = binary_predictions.size
    manual_bitwise_accuracy = correct_bits / total_bits
    print(f"Overall Bitwise Accuracy: {manual_bitwise_accuracy:.4f}")
    
    # Calculate accuracy per output bit position
    accuracies_per_bit = np.mean(binary_predictions == y_test[:len(binary_predictions)].reshape(binary_predictions.shape), axis=0)
    print(f"Mean accuracy across bit positions: {np.mean(accuracies_per_bit):.4f}")
    print(f"Min accuracy for a bit position: {np.min(accuracies_per_bit):.4f} (position {np.argmin(accuracies_per_bit)})")
    print(f"Max accuracy for a bit position: {np.max(accuracies_per_bit):.4f} (position {np.argmax(accuracies_per_bit)})")
    
    # Plot bit position accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(accuracies_per_bit)), accuracies_per_bit)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Bit Position')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Bit Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig("bit_position_accuracy.png", dpi=300)
    print("Bit position accuracy plot saved as bit_position_accuracy.png")
    
    # Calculate block prediction accuracy
    perfect_predictions = np.sum(np.all(binary_predictions == y_test[:len(binary_predictions)].reshape(binary_predictions.shape), axis=1))
    total_blocks = binary_predictions.shape[0]
    block_accuracy = perfect_predictions / total_blocks
    print(f"Perfect Block Prediction: {perfect_predictions}/{total_blocks} = {block_accuracy:.6f}")
    
    # Calculate distribution of correct bits per block
    correct_bits_per_block = np.sum(binary_predictions == y_test[:len(binary_predictions)].reshape(binary_predictions.shape), axis=1)
    plt.figure(figsize=(12, 6))
    plt.hist(correct_bits_per_block, bins=30, alpha=0.7)
    plt.axvline(x=128, color='r', linestyle='--', label='Perfect prediction (128 bits)')
    plt.axvline(x=64, color='g', linestyle='--', label='Random guessing (64 bits)')
    plt.xlabel('Number of Correct Bits per Block')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correct Bits per Block')
    plt.legend()
    plt.tight_layout()
    plt.savefig("correct_bits_distribution.png", dpi=300)
    print("Correct bits distribution plot saved as correct_bits_distribution.png")
    
    return binary_predictions, accuracies_per_bit, correct_bits_per_block

# --- Main execution ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Enhanced AES Neural Network Training")
    print("="*50)
    
    # Check for GPU and optimize
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). TensorFlow will use GPU: {gpus}")
            
            # Verify GPU is being used
            physical_devices = tf.config.list_physical_devices('GPU')
            details = tf.config.experimental.get_device_details(physical_devices[0])
            print(f"GPU details: {details}")
            
            # Set memory limit (adjust based on your RTX 3070's VRAM)
            # For RTX 3070 with 8GB VRAM, leave some memory for system
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]  # 7GB limit
            # )
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found. TensorFlow will use CPU.")

    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")

    # 1. Generate Data with multiprocessing
    print("\n" + "-"*50)
    print("Data Generation")
    print("-"*50)
    
    try:
        X, y = generate_dataset_parallel(NUM_SAMPLES)
        print(f"Generated dataset with X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"Error with multiprocessing: {e}")
        print("Using reduced dataset size with sequential processing...")
        # Use a smaller dataset size if parallel processing fails
        reduced_samples = 500000  # Reduce to a more manageable size
        X, y = generate_dataset_sequential(reduced_samples)
        print(f"Generated dataset with X shape: {X.shape}, y shape: {y.shape}")

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Release memory from original arrays
    del X, y, X_temp, y_temp
    
    # Convert to tf.data.Dataset with optimizations
    print("\n" + "-"*50)
    print("Preparing TensorFlow Datasets")
    print("-"*50)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=min(100000, X_train.shape[0]), seed=SEED)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    print("tf.data.Dataset pipelines optimized.")

    # 2. Create Enhanced Model
    print("\n" + "-"*50)
    print(f"Creating Enhanced Model (Residual Connections: {USE_RESIDUAL_CONNECTIONS})")
    print("-"*50)
    
    model = create_enhanced_model(input_shape=(X_train.shape[1],), output_shape=y_train.shape[1])
    model.summary()
    
    # Calculate model size
    model_params = model.count_params()
    print(f"Model has {model_params:,} parameters")

    # 3. Define Advanced Callbacks
    print("\n" + "-"*50)
    print("Setting up Training Callbacks")
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
        # Early stopping with increased patience
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        
        # Model checkpoint to save best models
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_CHECKPOINT_PATH, 'aes_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1,
            save_weights_only=False
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        ),
        
        # Reduce learning rate on plateau as backup
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # CSV logger to keep track of metrics
        callbacks.CSVLogger('training_log.csv', append=True)
    ]

    # 4. Training with GPU optimization
    print("\n" + "-"*50)
    print("Starting Enhanced Training")
    print("-"*50)
    
    # Report GPU memory before training
    if gpus:
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        for device in devices:
            if device.device_type == 'GPU':
                print(f"GPU Memory: {device.memory_limit / (1024**3):.2f} GB")
    
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

    # Plot training history
    plot_training_history(history)

    # 5. Evaluation and Analysis
    print("\n" + "-"*50)
    print("Detailed Model Evaluation")
    print("-"*50)
    
    # Load the best model
    best_model_path = os.path.join(MODEL_CHECKPOINT_PATH, sorted([f for f in os.listdir(MODEL_CHECKPOINT_PATH) if f.endswith('.keras')])[-1])
    print(f"Loading best model from: {best_model_path}")
    best_model = keras.models.load_model(best_model_path)
    
    # Evaluate on test set
    metrics = best_model.evaluate(test_dataset, verbose=1)
    metrics_names = best_model.metrics_names
    
    print("\nTest Metrics:")
    for name, value in zip(metrics_names, metrics):
        print(f"  {name}: {value:.4f}")
    
    # Detailed prediction analysis
    binary_predictions, accuracies_per_bit, correct_bits_per_block = analyze_predictions(best_model, test_dataset, y_test)

    # 6. Save the final model
    print("\n" + "-"*50)
    print("Saving Final Model")
    print("-"*50)
    
    final_model_path = "aes_nn_final_model.keras"
    best_model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\n" + "="*50)
    print("AES Neural Network Training Complete")
    print("="*50)
    print("\nNote: Learning AES is an extremely challenging task.")
    print("Even with these optimizations, perfect accuracy is highly unlikely.")
    print("The residual network architecture and larger dataset should provide better results than the original implementation.")

