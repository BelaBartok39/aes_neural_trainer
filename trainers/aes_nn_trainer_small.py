\
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from Crypto.Cipher import AES
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# --- Configuration ---
NUM_SAMPLES = 500000  # Increased number of samples
# For a real attempt to learn AES, a much larger dataset would be needed.
# This is a demonstrative size.
EPOCHS = 50  # Increased number of training epochs
BATCH_SIZE = 64  # Batch size for training (can also be tuned, e.g., 32, 128)
SEED = 42  # Random seed for reproducibility

# Set random seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)

# For tf.data API
AUTOTUNE = tf.data.AUTOTUNE

# --- 1. Data Generation ---
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

def generate_dataset(num_samples):
    """
    Generates a dataset of (plaintext, key, ciphertext) tuples.
    X consists of concatenated [plaintext_bits, key_bits].
    y consists of ciphertext_bits.
    """
    X_data = []
    y_data = []
    print(f"Generating {num_samples} samples...")
    start_time = time.time()
    # Initialize progress tracking for larger datasets
    if num_samples >= 10:
        progress_interval = num_samples // 10
    else:
        progress_interval = 1 # Avoid division by zero if num_samples is small

    for i in range(num_samples):
        pt_bits, key_bits, ct_bits = generate_sample()
        # Concatenate plaintext and key bits for the input features
        X_data.append(np.concatenate([pt_bits, key_bits]))
        y_data.append(ct_bits)
        if progress_interval > 0 and (i + 1) % progress_interval == 0 and num_samples >= 10:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds.")
    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

# --- 2. Model Construction ---
def create_model(input_shape=(256,), output_shape=128):
    """
    Defines and compiles the neural network model.
    Input: 256 units (128 for plaintext + 128 for key)
    Hidden layers: 3 dense layers with ReLU, 1024 units each, with Batch Norm and Dropout
    Output layer: 128 units with sigmoid (for bitwise binary output)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        
        layers.Dense(1024, name="hidden_layer_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Activation('relu', name="activation_1"),
        layers.Dropout(0.3, name="dropout_1"), # Dropout rate of 30%
        
        layers.Dense(1024, name="hidden_layer_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Activation('relu', name="activation_2"),
        layers.Dropout(0.3, name="dropout_2"),
        
        layers.Dense(512, name="hidden_layer_3"), # Added a third, slightly smaller hidden layer
        layers.BatchNormalization(name="bn_3"),
        layers.Activation('relu', name="activation_3"),
        layers.Dropout(0.3, name="dropout_3"),
        
        layers.Dense(output_shape, activation='sigmoid', name="output_layer")
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']) # 'accuracy' here is bitwise accuracy
    return model

# --- Utility function for plotting training history ---
def plot_training_history(history):
    """Plots training and validation accuracy and loss."""
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training history plot saved as training_history.png")
    # plt.show() # Uncomment to display the plot directly if running in an interactive environment

# --- Main execution ---
if __name__ == "__main__":
    print("Starting AES Neural Network Learning Script...")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be set per GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). TensorFlow will use: {gpus}")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Error setting memory growth for GPU: {e}")
    else:
        print("No GPU found. TensorFlow will use CPU.")

    print(f"Using TensorFlow version: {tf.__version__}")

    # 1. Generate Data
    X, y = generate_dataset(NUM_SAMPLES)
    print(f"Generated dataset with X shape: {X.shape}, y shape: {y.shape}")

    # Split dataset into training, validation, and test sets
    # 70% training, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Convert to tf.data.Dataset for efficient training
    print("\n--- Preparing tf.data.Dataset pipelines ---")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache() # Cache dataset in memory after first epoch
    train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0], seed=SEED) # Shuffle data
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE) # Prefetch data for GPU

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE) # No need to cache or shuffle test set usually
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    print("tf.data.Dataset pipelines prepared.")

    # 2. Create and Compile Model
    model = create_model(input_shape=(X_train.shape[1],), output_shape=y_train.shape[1])
    model.summary()

    # 3. Training
    print("\\n--- Training ---")
    
    # Callbacks for training
    # Stop training if validation loss doesn't improve for 'patience' epochs
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Increased patience slightly
    # Save the model checkpoint (optional, but good practice)
    model_checkpoint = callbacks.ModelCheckpoint('aes_nn_model_best.keras', save_best_only=True, monitor='val_loss')

    start_time = time.time()
    history = model.fit(train_dataset, # Use tf.data.Dataset here
                        epochs=EPOCHS,
                        batch_size=None, # Batching is handled by the dataset
                        validation_data=val_dataset, # Use tf.data.Dataset here
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Plot training history
    plot_training_history(history)

    # 4. Evaluation
    print("\\n--- Evaluation on Test Set ---")
    # Load the best model saved by ModelCheckpoint if you want to be sure
    # model = keras.models.load_model('aes_nn_model_best.keras')
    
    loss, accuracy = model.evaluate(test_dataset, verbose=0) # Use tf.data.Dataset here
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Bitwise Accuracy: {accuracy:.4f} (This is the average accuracy per bit)")

    # Optional: Analyze bitwise accuracy in more detail
    # For model.predict, it can be more efficient to predict on the full X_test numpy array
    # or ensure the test_dataset for predict is not repeating/shuffling if not intended.
    # The current test_dataset is fine for predict as it's just batched.
    print("\n--- Predicting on Test Set for detailed analysis ---")
    predictions = model.predict(test_dataset) # Use tf.data.Dataset here
    # Convert sigmoid outputs (probabilities) to binary predictions (0 or 1)
    # Need to handle the fact that predictions might be on a batched dataset.
    # However, model.predict with a tf.data.Dataset will return a single NumPy array
    # if all batches are processed. Let's assume y_test is still the original NumPy array for comparison.
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate overall bitwise accuracy manually (should match model.evaluate's accuracy)
    # Ensure y_test corresponds to the order of predictions if test_dataset was shuffled (it wasn't here)
    correct_bits = np.sum(binary_predictions == y_test[:binary_predictions.shape[0]*binary_predictions.shape[1]].reshape(binary_predictions.shape)) # Adjust y_test to match potentially batched prediction output size
    total_bits = binary_predictions.size
    manual_bitwise_accuracy = correct_bits / total_bits
    print(f"Manual Overall Bitwise Accuracy: {manual_bitwise_accuracy:.4f}")

    # Calculate accuracy per output bit position (to see if some bits are easier to predict)
    accuracies_per_bit = np.mean(binary_predictions == y_test[:binary_predictions.shape[0]*binary_predictions.shape[1]].reshape(binary_predictions.shape), axis=0)
    print(f"Mean accuracy across all bit positions: {np.mean(accuracies_per_bit):.4f}")
    print(f"Min accuracy for a single bit position: {np.min(accuracies_per_bit):.4f}")
    print(f"Max accuracy for a single bit position: {np.max(accuracies_per_bit):.4f}")

    # A more challenging metric: Perfect block prediction accuracy
    # How many 128-bit ciphertext blocks were predicted perfectly?
    perfect_predictions = np.sum(np.all(binary_predictions == y_test[:binary_predictions.shape[0]*binary_predictions.shape[1]].reshape(binary_predictions.shape), axis=1))
    total_blocks = binary_predictions.shape[0] # Number of blocks in predictions
    block_accuracy = perfect_predictions / total_blocks
    print(f"Perfect Block Prediction Accuracy: {perfect_predictions}/{total_blocks} = {block_accuracy:.4f}")
    
    print("\\n--- Script Finished ---")
    print("Note: Learning AES with a neural network is an extremely hard problem.")
    print("Achieving high accuracy, especially perfect block prediction, is highly unlikely with this setup.")
    print("This script serves as a framework for experimentation.")

# To run this script:
# 1. Make sure you have Python installed.
# 2. Install necessary libraries:
#    pip install numpy tensorflow pycryptodome scikit-learn matplotlib
# 3. Save this code as a .py file (e.g., aes_nn_trainer.py)
# 4. Run from your terminal: python aes_nn_trainer.py

