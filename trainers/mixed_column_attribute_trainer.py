import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import seaborn as sns

# Configuration
SEED = 42
NUM_SAMPLES = 100000
BATCH_SIZE = 256
EPOCHS = 100
OUTPUT_DIR = "galois_field_learning"
USE_EARLY_STOPPING = True
PATIENCE = 20

# Set random seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Galois Field Operations
# -------------------------------------------------------------------------

def galois_multiply(a, b):
    """Multiply two numbers in GF(2^8)"""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        high_bit_set = a & 0x80
        a <<= 1
        if high_bit_set:
            a ^= 0x1B  # AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1
        b >>= 1
    return p & 0xFF

# Create lookup tables for Galois field multiplication
GF_MUL_2 = np.array([galois_multiply(i, 2) for i in range(256)], dtype=np.uint8)
GF_MUL_3 = np.array([galois_multiply(i, 3) for i in range(256)], dtype=np.uint8)

# -------------------------------------------------------------------------
# Dataset Generation
# -------------------------------------------------------------------------

def generate_gf_multiply_dataset(num_samples, multiplier=2):
    """Generate dataset for learning GF(2^8) multiplication by a constant"""
    # Choose the appropriate table
    if multiplier == 2:
        lookup_table = GF_MUL_2
    elif multiplier == 3:
        lookup_table = GF_MUL_3
    else:
        lookup_table = np.array([galois_multiply(i, multiplier) for i in range(256)], dtype=np.uint8)
    
    # Generate random inputs
    X = np.random.randint(0, 256, size=(num_samples, 1), dtype=np.uint8)
    y = np.zeros_like(X)
    
    # Calculate outputs using the lookup table
    for i in range(num_samples):
        y[i, 0] = lookup_table[X[i, 0]]
    
    # Convert to bit representation - multiple formats for experimentation
    
    # 1. Bit vector representation (8 bits per input/output)
    X_bits = np.unpackbits(X, axis=1)
    y_bits = np.unpackbits(y, axis=1)
    
    # 2. Normalized byte representation (0-255 -> 0-1)
    X_norm = X.astype(np.float32) / 255.0
    y_norm = y.astype(np.float32) / 255.0
    
    # 3. One-hot encoded representation (256 classes)
    X_onehot = tf.keras.utils.to_categorical(X, num_classes=256)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=256)
    
    # 4. Concatenate the original X with binary operations results for structure hints
    # For example, include X << 1 as an additional feature to help with learning GF(2^8) x 2
    if multiplier == 2:
        # For multiplication by 2, left shift and conditional XOR with 0x1B are relevant
        X_shifted = (X << 1) & 0xFF  # Left shift
        X_xor = np.where((X & 0x80) > 0, 0x1B, 0)  # Conditional XOR with polynomial
        
        # Stack features
        X_structured = np.concatenate([
            X, 
            X_shifted, 
            X_xor,
            (X_shifted ^ X_xor).astype(np.uint8)  # The expected result
        ], axis=1).astype(np.float32) / 255.0
    else:
        # For other multipliers, include relevant operations
        X_structured = np.concatenate([
            X,
            GF_MUL_2.reshape(-1, 1)[X].reshape(num_samples, 1)  # Add mul by 2 result
        ], axis=1).astype(np.float32) / 255.0
    
    return {
        'X_raw': X,
        'y_raw': y,
        'X_bits': X_bits.astype(np.float32),
        'y_bits': y_bits.astype(np.float32),
        'X_norm': X_norm,
        'y_norm': y_norm,
        'X_onehot': X_onehot,
        'y_onehot': y_onehot,
        'X_structured': X_structured,
        'y_structured': y_norm  # Same as y_norm for simplicity
    }

def generate_full_mixcolumns_dataset(num_samples):
    """Generate dataset for learning the full MixColumns operation"""
    # MixColumns operates on 4x4 state
    X = np.random.randint(0, 256, size=(num_samples, 16), dtype=np.uint8)
    y = np.zeros_like(X)
    
    for i in range(num_samples):
        # Reshape to 4x4 state (column-major order for AES)
        state = np.zeros((4, 4), dtype=np.uint8)
        for col in range(4):
            for row in range(4):
                state[row, col] = X[i, col * 4 + row]
        
        # Apply MixColumns to each column
        result_state = np.zeros_like(state)
        for col in range(4):
            # MixColumns matrix multiplication
            result_state[0, col] = (galois_multiply(state[0, col], 2) ^ 
                                   galois_multiply(state[1, col], 3) ^ 
                                   state[2, col] ^ 
                                   state[3, col])
            
            result_state[1, col] = (state[0, col] ^ 
                                   galois_multiply(state[1, col], 2) ^ 
                                   galois_multiply(state[2, col], 3) ^ 
                                   state[3, col])
            
            result_state[2, col] = (state[0, col] ^ 
                                   state[1, col] ^ 
                                   galois_multiply(state[2, col], 2) ^ 
                                   galois_multiply(state[3, col], 3))
            
            result_state[3, col] = (galois_multiply(state[0, col], 3) ^ 
                                   state[1, col] ^ 
                                   state[2, col] ^ 
                                   galois_multiply(state[3, col], 2))
        
        # Flatten back to 1D array
        for col in range(4):
            for row in range(4):
                y[i, col * 4 + row] = result_state[row, col]
    
    # Convert to bit representation
    X_bits = np.unpackbits(X, axis=1)
    y_bits = np.unpackbits(y, axis=1)
    
    # Normalized representation
    X_norm = X.astype(np.float32) / 255.0
    y_norm = y.astype(np.float32) / 255.0
    
    return {
        'X_raw': X,
        'y_raw': y,
        'X_bits': X_bits.astype(np.float32),
        'y_bits': y_bits.astype(np.float32),
        'X_norm': X_norm,
        'y_norm': y_norm
    }

def split_dataset(dataset, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test sets"""
    num_samples = len(dataset['X_raw'])
    indices = np.random.permutation(num_samples)
    
    test_size = int(num_samples * test_ratio)
    val_size = int(num_samples * val_ratio)
    train_size = num_samples - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_data = {}
    val_data = {}
    test_data = {}
    
    for key in dataset.keys():
        train_data[key] = dataset[key][train_indices]
        val_data[key] = dataset[key][val_indices]
        test_data[key] = dataset[key][test_indices]
    
    return train_data, val_data, test_data

# -------------------------------------------------------------------------
# Custom Layers and Models
# -------------------------------------------------------------------------

class BinaryConv2D(layers.Layer):
    """Binary Convolutional Layer with XOR-like operations"""
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super(BinaryConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
    
    def build(self, input_shape):
        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # Binarize inputs and weights
        bin_inputs = K.sign(K.clip(inputs, -1, 1))
        bin_kernel = K.sign(K.clip(self.kernel, -1, 1))
        
        # Convolve with binary weights
        output = K.conv2d(
            bin_inputs,
            bin_kernel,
            strides=self.strides,
            padding=self.padding
        )
        return output

class GaloisMul2Layer(layers.Layer):
    """Layer that learns GF(2^8) multiplication by 2"""
    def __init__(self, **kwargs):
        super(GaloisMul2Layer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable lookup table
        self.lookup = self.add_weight(
            name='gf_mul2_lookup',
            shape=(256,),
            initializer=keras.initializers.Constant(GF_MUL_2 / 255.0),  # Initialize with correct values
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # Scale inputs from [0,1] to integers
        x_scaled = tf.cast(inputs * 255.0, tf.int32)
        x_clipped = tf.clip_by_value(x_scaled, 0, 255)
        
        # Lookup the result in our table
        return tf.gather(self.lookup, x_clipped)

class GaloisMul3Layer(layers.Layer):
    """Layer that learns GF(2^8) multiplication by 3"""
    def __init__(self, **kwargs):
        super(GaloisMul3Layer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable lookup table
        self.lookup = self.add_weight(
            name='gf_mul3_lookup',
            shape=(256,),
            initializer=keras.initializers.Constant(GF_MUL_3 / 255.0),  # Initialize with correct values
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # Scale inputs from [0,1] to integers
        x_scaled = tf.cast(inputs * 255.0, tf.int32)
        x_clipped = tf.clip_by_value(x_scaled, 0, 255)
        
        # Lookup the result in our table
        return tf.gather(self.lookup, x_clipped)

class BitXORLayer(layers.Layer):
    """Layer that approximates bitwise XOR operation"""
    def __init__(self, **kwargs):
        super(BitXORLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # No weights needed
        self.built = True
    
    def call(self, inputs):
        # Assumes inputs is a list of two tensors with identical shapes
        x1, x2 = inputs
        
        # Scale to [0,1] range if not already
        x1 = tf.clip_by_value(x1, 0, 1)
        x2 = tf.clip_by_value(x2, 0, 1)
        
        # Approximate XOR: (x1 * (1 - x2)) + (x2 * (1 - x1))
        return (x1 * (1 - x2)) + (x2 * (1 - x1))

# -------------------------------------------------------------------------
# Neural Network Models
# -------------------------------------------------------------------------

def create_binary_mlp(input_shape, output_shape):
    """Create a binary MLP for learning bit operations"""
    inputs = layers.Input(shape=input_shape)
    
    # Reshape inputs to be -1 to 1 (for binary operations)
    x = layers.Lambda(lambda x: x * 2 - 1)(inputs)
    
    # Dense layers with binarization
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda x: K.sign(K.clip(x, -1, 1)))(x)
    
    x = layers.Dense(256)(x)
    x = layers.Lambda(lambda x: K.sign(K.clip(x, -1, 1)))(x)
    
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda x: K.sign(K.clip(x, -1, 1)))(x)
    
    # Output layer
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def create_structured_gf2_model():
    """Create a model with explicit structure for GF(2^8) multiplication by 2"""
    inputs = layers.Input(shape=(1,))
    
    # Scale input from 0-1 to 0-255
    x = layers.Lambda(lambda x: x * 255.0)(inputs)
    
    # Extract the highest bit (x >= 128)
    high_bit = layers.Lambda(lambda x: tf.cast(x >= 128.0, tf.float32))(x)
    
    # Left shift by 1 (multiply by 2)
    shifted = layers.Lambda(lambda x: tf.math.mod(x * 2.0, 256.0))(x)
    
    # Conditional XOR with 0x1B if high bit is set
    xor_val = layers.Lambda(lambda x: x * 0x1B)(high_bit)
    
    # Combine with XOR
    result = layers.Lambda(lambda x: tf.math.mod(x[0] + x[1] - 2 * x[0] * x[1], 256.0))([shifted, xor_val])
    
    # Scale back to 0-1
    outputs = layers.Lambda(lambda x: x / 255.0)(result)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def create_mixcolumns_component_model():
    """Create a model that uses separate components for MixColumns operation"""
    inputs = layers.Input(shape=(16,))  # 16 bytes of state
    
    # Reshape to 4x4 state (column-major order for AES)
    state = layers.Reshape((4, 4, 1))(inputs)
    
    # Process each column independently
    outputs = []
    for col in range(4):
        col_input = layers.Lambda(lambda x: x[:, :, col, :])(state)
        
        # Apply MixColumns matrix multiplication component by component
        # Component layers
        s0 = layers.Lambda(lambda x: x[:, 0, 0])(col_input)
        s1 = layers.Lambda(lambda x: x[:, 1, 0])(col_input)
        s2 = layers.Lambda(lambda x: x[:, 2, 0])(col_input)
        s3 = layers.Lambda(lambda x: x[:, 3, 0])(col_input)
        
        # Apply Galois field multiplications
        # Row 0: (2*s0) ⊕ (3*s1) ⊕ s2 ⊕ s3
        mul2_s0 = GaloisMul2Layer()(s0)
        mul3_s1 = GaloisMul3Layer()(s1)
        row0 = BitXORLayer()([mul2_s0, mul3_s1])
        row0 = BitXORLayer()([row0, s2])
        row0 = BitXORLayer()([row0, s3])
        
        # Row 1: s0 ⊕ (2*s1) ⊕ (3*s2) ⊕ s3
        mul2_s1 = GaloisMul2Layer()(s1)
        mul3_s2 = GaloisMul3Layer()(s2)
        row1 = BitXORLayer()([s0, mul2_s1])
        row1 = BitXORLayer()([row1, mul3_s2])
        row1 = BitXORLayer()([row1, s3])
        
        # Row 2: s0 ⊕ s1 ⊕ (2*s2) ⊕ (3*s3)
        mul2_s2 = GaloisMul2Layer()(s2)
        mul3_s3 = GaloisMul3Layer()(s3)
        row2 = BitXORLayer()([s0, s1])
        row2 = BitXORLayer()([row2, mul2_s2])
        row2 = BitXORLayer()([row2, mul3_s3])
        
        # Row 3: (3*s0) ⊕ s1 ⊕ s2 ⊕ (2*s3)
        mul3_s0 = GaloisMul3Layer()(s0)
        mul2_s3 = GaloisMul2Layer()(s3)
        row3 = BitXORLayer()([mul3_s0, s1])
        row3 = BitXORLayer()([row3, s2])
        row3 = BitXORLayer()([row3, mul2_s3])
        
        # Stack the results for this column
        col_output = layers.Lambda(lambda x: tf.stack([x[0], x[1], x[2], x[3]], axis=1))(
            [row0, row1, row2, row3])
        
        outputs.append(col_output)
    
    # Combine all columns
    combined = layers.Lambda(lambda x: tf.stack(x, axis=2))(outputs)
    
    # Reshape back to 16 bytes
    final_output = layers.Reshape((16,))(combined)
    
    return keras.Model(inputs=inputs, outputs=final_output)

# -------------------------------------------------------------------------
# Custom Loss Functions
# -------------------------------------------------------------------------

def bit_level_accuracy(y_true, y_pred):
    """Calculate accuracy at the bit level"""
    # Convert predictions to binary
    y_pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    
    # Calculate bit accuracy
    correct_bits = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32))
    return correct_bits

def byte_level_accuracy(y_true, y_pred):
    """Calculate accuracy at the byte level (all 8 bits must be correct)"""
    # Reshape to have 8 bits per byte
    y_true_reshaped = tf.reshape(y_true, (-1, 8))
    y_pred_reshaped = tf.reshape(y_pred, (-1, 8))
    
    # Convert predictions to binary
    y_pred_binary = tf.cast(tf.greater_equal(y_pred_reshaped, 0.5), tf.float32)
    
    # Check if all bits in a byte match
    correct_bytes = tf.reduce_all(tf.equal(y_true_reshaped, y_pred_binary), axis=1)
    
    # Calculate byte accuracy
    return tf.reduce_mean(tf.cast(correct_bytes, tf.float32))

def hamming_loss(y_true, y_pred):
    """Calculate Hamming distance (number of bit errors)"""
    # Convert predictions to binary
    y_pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    
    # Count differences
    hamming_dist = tf.reduce_mean(tf.cast(tf.not_equal(y_true, y_pred_binary), tf.float32))
    return hamming_dist

def balanced_binary_crossentropy(y_true, y_pred):
    """Binary crossentropy that balances 0s and 1s (since bit patterns may be imbalanced)"""
    # Compute the number of 1s and 0s in the true labels
    num_pos = tf.reduce_sum(y_true)
    num_neg = tf.reduce_sum(1.0 - y_true)
    
    # Compute weights for positive and negative classes
    pos_weight = num_neg / (num_pos + num_neg)
    neg_weight = num_pos / (num_pos + num_neg)
    
    # Apply weighted binary crossentropy
    pos_loss = -pos_weight * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
    neg_loss = -neg_weight * (1.0 - y_true) * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-7, 1.0))
    
    return tf.reduce_mean(pos_loss + neg_loss)

# -------------------------------------------------------------------------
# Training and Evaluation Functions
# -------------------------------------------------------------------------

def train_model(model, train_data, val_data, input_key='X_bits', output_key='y_bits', 
                loss='binary_crossentropy', metrics=None, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """Train a model with the specified data and parameters"""
    # Configure metrics
    if metrics is None:
        metrics = ['accuracy', bit_level_accuracy, byte_level_accuracy]
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=metrics
    )
    
    # Callbacks
    callbacks = []
    if USE_EARLY_STOPPING:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True
            )
        )
    
    # Add TensorBoard logging
    log_dir = os.path.join(OUTPUT_DIR, f"logs_{int(time.time())}")
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        train_data[input_key], train_data[output_key],
        validation_data=(val_data[input_key], val_data[output_key]),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Print training time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")
    
    return model, history

def evaluate_model(model, test_data, input_key='X_bits', output_key='y_bits'):
    """Evaluate model performance on test data"""
    # Get predictions
    y_pred = model.predict(test_data[input_key])
    
    # Calculate metrics
    test_loss, test_acc = model.evaluate(test_data[input_key], test_data[output_key], verbose=0)
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # For bit representation, calculate bit-level metrics
    if 'bits' in output_key:
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        # Calculate bit-level accuracy
        bit_accuracies = []
        for i in range(test_data[output_key].shape[1]):
            bit_acc = np.mean(y_pred_binary[:, i] == test_data[output_key][:, i])
            bit_accuracies.append(bit_acc)
        
        avg_bit_acc = np.mean(bit_accuracies)
        min_bit_acc = np.min(bit_accuracies)
        max_bit_acc = np.max(bit_accuracies)
        
        print(f"Average bit accuracy: {avg_bit_acc:.4f}")
        print(f"Minimum bit accuracy: {min_bit_acc:.4f}")
        print(f"Maximum bit accuracy: {max_bit_acc:.4f}")
        
        # Plot bit accuracies
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(bit_accuracies)), bit_accuracies)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
        plt.xlabel('Bit Position')
        plt.ylabel('Accuracy')
        plt.title('Bit-level Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'bit_accuracies.png'), dpi=300)
        plt.close()
        
        # Calculate byte-level accuracy (if applicable)
        if len(bit_accuracies) % 8 == 0:
            num_bytes = len(bit_accuracies) // 8
            byte_correct = 0
            total_bytes = len(test_data[output_key])
            
            for i in range(total_bytes):
                for b in range(num_bytes):
                    start_bit = b * 8
                    end_bit = (b + 1) * 8
                    byte_pred = y_pred_binary[i, start_bit:end_bit]
                    byte_true = test_data[output_key][i, start_bit:end_bit]
                    if np.array_equal(byte_pred, byte_true):
                        byte_correct += 1
            
            byte_accuracy = byte_correct / (total_bytes * num_bytes)
            print(f"Byte-level accuracy: {byte_accuracy:.4f}")
    
    # For normalized representation, calculate value-based metrics
    elif 'norm' in output_key:
        # Convert predictions to 0-255 range
        y_pred_scaled = np.round(y_pred * 255).astype(np.uint8)
        y_true_scaled = np.round(test_data[output_key] * 255).astype(np.uint8)
        
        # Calculate exact match accuracy
        exact_matches = np.mean(y_pred_scaled == y_true_scaled)
        print(f"Exact value match accuracy: {exact_matches:.4f}")
        
        # Calculate average error
        avg_error = np.mean(np.abs(y_pred_scaled - y_true_scaled))
        print(f"Average error magnitude: {avg_error:.2f}")
        
        # Plot error distribution
        errors = np.abs(y_pred_scaled - y_true_scaled).flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=range(257), alpha=0.7)
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution.png'), dpi=300)
        plt.close()
    
    return y_pred

# -------------------------------------------------------------------------
# Specific Analysis Functions
# -------------------------------------------------------------------------

def analyze_gf_multiplication(model, multiplier=2):
    """Analyze how well the model learned GF(2^8) multiplication"""
    print(f"\nAnalyzing GF(2^8) multiplication by {multiplier}...")
    
    # Generate all possible inputs (0-255)
    all_inputs = np.arange(256).reshape(-1, 1)
    
    # Prepare inputs for the model
    if hasattr(model, 'input_shape') and model.input_shape[1] == 8:
        # Binary representation
        model_inputs = np.unpackbits(all_inputs, axis=1).astype(np.float32)
    else:
        # Normalized representation
        model_inputs = all_inputs.astype(np.float32) / 255.0
    
    # Get model predictions
    predictions = model.predict(model_inputs)
    
    # Convert predictions to values
    if predictions.shape[1] == 8:
        # Binary representation
        pred_binary = (predictions > 0.5).astype(np.uint8)
        pred_values = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            pred_values[i] = np.packbits(pred_binary[i])[0]
    else:
        # Normalized representation
        pred_values = np.round(predictions.flatten() * 255).astype(np.uint8)
    
    # Calculate expected values
    if multiplier == 2:
        expected_values = GF_MUL_2
    elif multiplier == 3:
        expected_values = GF_MUL_3
    else:
        expected_values = np.array([galois_multiply(i, multiplier) for i in range(256)], dtype=np.uint8)
    
    # Calculate accuracy
    correct = (pred_values == expected_values)
    accuracy = np.mean(correct)
    
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Correctly predicted: {np.sum(correct)}/256 values")
    
    # Plot comparison for a sample of values
    sample_size = min(50, 256)
    indices = np.random.choice(256, sample_size, replace=False)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(all_inputs[indices], expected_values[indices], label='Expected', color='blue', alpha=0.7)
    plt.scatter(all_inputs[indices], pred_values[indices], label='Predicted', color='red', marker='x')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title(f'GF(2^8) Multiplication by {multiplier}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_comparison.png'), dpi=300)
    plt.close()
    
    # Plot all values comparison
    plt.figure(figsize=(12, 6))
    plt.plot(all_inputs, expected_values, label='Expected')
    plt.plot(all_inputs, pred_values, label='Predicted', alpha=0.7)
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Full Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_full_comparison.png'), dpi=300)
    plt.close()
    
    # Plot error distribution
    errors = np.abs(pred_values - expected_values)
    avg_error = np.mean(errors)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=range(257), alpha=0.7)
    plt.axvline(x=avg_error, color='r', linestyle='--', label=f'Average error: {avg_error:.2f}')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_error_distribution.png'), dpi=300)
    plt.close()
    
    # Analyze bit-level accuracy
    expected_bits = np.unpackbits(expected_values.reshape(-1, 1), axis=1)
    pred_bits = np.unpackbits(pred_values.reshape(-1, 1), axis=1)
    
    bit_accuracies = []
    for i in range(8):
        bit_acc = np.mean(expected_bits[:, i] == pred_bits[:, i])
        bit_accuracies.append(bit_acc)
    
    print("\nBit-level accuracy:")
    for i, acc in enumerate(bit_accuracies):
        print(f"Bit {i}: {acc:.4f}")
    
    # Plot bit-level accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(range(8), bit_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Bit Position')
    plt.ylabel('Accuracy')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Bit-level Accuracy')
    plt.xticks(range(8), [f'Bit {i}' for i in range(8)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_bit_accuracy.png'), dpi=300)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'errors': errors,
        'bit_accuracies': bit_accuracies
    }

def analyze_mixcolumns_performance(model, test_data):
    """Analyze how well the model learned the MixColumns transformation"""
    print("\nAnalyzing MixColumns performance...")
    
    # Get predictions
    X_norm = test_data['X_norm'][:100]  # Use a subset for analysis
    y_norm = test_data['y_norm'][:100]
    X_raw = test_data['X_raw'][:100]
    y_raw = test_data['y_raw'][:100]
    
    # Get model predictions
    y_pred = model.predict(X_norm)
    
    # Convert predictions to bytes
    y_pred_bytes = np.round(y_pred * 255).astype(np.uint8)
    
    # Calculate byte-level accuracy
    byte_correct = np.sum(y_pred_bytes == y_raw)
    byte_total = y_raw.size
    byte_accuracy = byte_correct / byte_total
    
    print(f"Byte-level accuracy: {byte_accuracy:.4f} ({byte_correct}/{byte_total} bytes)")
    
    # Calculate average error
    byte_errors = np.abs(y_pred_bytes - y_raw)
    avg_error = np.mean(byte_errors)
    
    print(f"Average error magnitude: {avg_error:.2f}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(byte_errors.flatten(), bins=range(257), alpha=0.7)
    plt.axvline(x=avg_error, color='r', linestyle='--', label=f'Average error: {avg_error:.2f}')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.title('MixColumns - Byte Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns_error_distribution.png'), dpi=300)
    plt.close()
    
    # Analyze column-wise performance
    column_accuracies = []
    for col in range(4):
        col_indices = [col * 4 + i for i in range(4)]
        col_correct = np.sum(y_pred_bytes[:, col_indices] == y_raw[:, col_indices])
        col_total = y_raw[:, col_indices].size
        col_acc = col_correct / col_total
        column_accuracies.append(col_acc)
        print(f"Column {col} accuracy: {col_acc:.4f}")
    
    # Visualize column accuracies
    plt.figure(figsize=(8, 6))
    plt.bar(range(4), column_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Column Index')
    plt.ylabel('Accuracy')
    plt.title('MixColumns - Column Accuracy')
    plt.xticks(range(4), [f'Column {i}' for i in range(4)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns_column_accuracy.png'), dpi=300)
    plt.close()
    
    return {
        'byte_accuracy': byte_accuracy,
        'avg_error': avg_error,
        'column_accuracies': column_accuracies
    }

# -------------------------------------------------------------------------
# Main Experiment Functions
# -------------------------------------------------------------------------

def experiment_gf_mul2():
    """Experiment with learning GF(2^8) multiplication by 2"""
    print("\n" + "="*80)
    print("Experiment: GF(2^8) Multiplication by 2")
    print("="*80)
    
    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, "gf_mul2")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate dataset
    dataset = generate_gf_multiply_dataset(NUM_SAMPLES, multiplier=2)
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Try different approaches
    
    # 1. Binary MLP with bit representation
    print("\nTraining Binary MLP with bit representation...")
    binary_mlp = create_binary_mlp(input_shape=(8,), output_shape=8)
    binary_mlp.summary()
    
    binary_mlp, history = train_model(
        binary_mlp, train_data, val_data, 
        input_key='X_bits', output_key='y_bits',
        loss=balanced_binary_crossentropy
    )
    
    # Evaluate
    evaluate_model(binary_mlp, test_data, input_key='X_bits', output_key='y_bits')
    
    # Analyze
    analyze_gf_multiplication(binary_mlp, multiplier=2)
    
    # Save model
    binary_mlp.save(os.path.join(exp_dir, "binary_mlp.h5"))
    
    # 2. Structured model with normalized representation
    print("\nTraining Structured GF(2) model...")
    structured_model = create_structured_gf2_model()
    structured_model.summary()
    
    structured_model, history = train_model(
        structured_model, train_data, val_data, 
        input_key='X_norm', output_key='y_norm',
        loss='mse'
    )
    
    # Evaluate
    evaluate_model(structured_model, test_data, input_key='X_norm', output_key='y_norm')
    
    # Analyze
    analyze_gf_multiplication(structured_model, multiplier=2)
    
    # Save model
    structured_model.save(os.path.join(exp_dir, "structured_model.h5"))
    
    return binary_mlp, structured_model

def experiment_gf_mul3():
    """Experiment with learning GF(2^8) multiplication by 3"""
    print("\n" + "="*80)
    print("Experiment: GF(2^8) Multiplication by 3")
    print("="*80)
    
    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, "gf_mul3")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate dataset
    dataset = generate_gf_multiply_dataset(NUM_SAMPLES, multiplier=3)
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Use the same model architectures as for GF(2^8) mul by 2
    
    # 1. Binary MLP with bit representation
    print("\nTraining Binary MLP with bit representation...")
    binary_mlp = create_binary_mlp(input_shape=(8,), output_shape=8)
    
    binary_mlp, history = train_model(
        binary_mlp, train_data, val_data, 
        input_key='X_bits', output_key='y_bits',
        loss=balanced_binary_crossentropy
    )
    
    # Evaluate
    evaluate_model(binary_mlp, test_data, input_key='X_bits', output_key='y_bits')
    
    # Analyze
    analyze_gf_multiplication(binary_mlp, multiplier=3)
    
    # Save model
    binary_mlp.save(os.path.join(exp_dir, "binary_mlp.h5"))
    
    # 2. More complex network for this operation
    print("\nTraining deeper network for GF(3) multiplication...")
    
    # Create a more complex model
    inputs = layers.Input(shape=(1,))
    x = layers.Lambda(lambda x: x * 255.0)(inputs)  # Scale to 0-255
    
    # First represent multiplication by 2
    x1 = layers.Dense(64, activation='relu')(x)
    x1 = layers.Dense(128, activation='relu')(x1)
    x1 = layers.Dense(64, activation='relu')(x1)
    mul2 = layers.Dense(1, activation='sigmoid')(x1)
    mul2 = layers.Lambda(lambda x: x * 255.0)(mul2)  # Scale back to 0-255
    
    # Then add original input (mul by 3 = mul by 2 + original)
    added = layers.Add()([mul2, x])
    result = layers.Lambda(lambda x: tf.math.mod(x, 256.0))(added)
    
    # Scale back to 0-1
    outputs = layers.Lambda(lambda x: x / 255.0)(result)
    
    complex_model = keras.Model(inputs=inputs, outputs=outputs)
    complex_model.summary()
    
    complex_model, history = train_model(
        complex_model, train_data, val_data, 
        input_key='X_norm', output_key='y_norm',
        loss='mse'
    )
    
    # Evaluate
    evaluate_model(complex_model, test_data, input_key='X_norm', output_key='y_norm')
    
    # Analyze
    analyze_gf_multiplication(complex_model, multiplier=3)
    
    # Save model
    complex_model.save(os.path.join(exp_dir, "complex_model.h5"))
    
    return binary_mlp, complex_model

def experiment_mixcolumns():
    """Experiment with learning the full MixColumns operation"""
    print("\n" + "="*80)
    print("Experiment: Full MixColumns Transformation")
    print("="*80)
    
    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, "mixcolumns")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate dataset
    dataset = generate_full_mixcolumns_dataset(NUM_SAMPLES)
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Try with component-based model
    print("\nTraining Component-based MixColumns model...")
    mixcol_model = create_mixcolumns_component_model()
    mixcol_model.summary()
    
    mixcol_model, history = train_model(
        mixcol_model, train_data, val_data, 
        input_key='X_norm', output_key='y_norm',
        loss='mse', epochs=200  # More epochs for this complex task
    )
    
    # Evaluate
    evaluate_model(mixcol_model, test_data, input_key='X_norm', output_key='y_norm')
    
    # Analyze
    results = analyze_mixcolumns_performance(mixcol_model, test_data)
    
    # Save model
    mixcol_model.save(os.path.join(exp_dir, "mixcolumns_model.h5"))
    
    return mixcol_model, results

def main():
    """Run the main experiment pipeline"""
    print("\n" + "="*80)
    print("Galois Field Neural Networks Framework")
    print("="*80)
    
    # Run experiments in sequence
    print("\nStarting with basic GF(2^8) multiplication experiments...")
    
    # Experiment 1: GF(2^8) multiplication by 2
    gf2_binary_mlp, gf2_structured = experiment_gf_mul2()
    
    # Experiment 2: GF(2^8) multiplication by 3
    gf3_binary_mlp, gf3_complex = experiment_gf_mul3()
    
    # Experiment 3: Full MixColumns with component-based model
    mixcol_model, mixcol_results = experiment_mixcolumns()
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

if __name__ == "__main__":
    main()

