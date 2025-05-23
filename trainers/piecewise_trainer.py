import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from Crypto.Cipher import AES
from Crypto.Util.strxor import strxor
import matplotlib.pyplot as plt
import time
import os
import mlflow
import mlflow.tensorflow
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import entropy
import pickle

# --- Configuration ---
SEED = 42
NUM_SAMPLES = 200000  # Number of samples per operation
BATCH_SIZE = 128  # Increased for A100 GPUs
EPOCHS = 100
OUTPUT_DIR = "aes_piecewise_analysis"
USE_EARLY_STOPPING = True
PATIENCE = 20
USE_MIXED_PRECISION = False
USE_XLA = True  # Enable XLA optimization
USE_DISTRIBUTED = False  # Enable distributed training

# AES operations to analyze
OPERATIONS = ["sbox", "shiftrows", "mixcolumns", "addroundkey", "full_round", "key_schedule"]
SELECTED_OPERATION = "mixcolumns"  # Change this to analyze different operations

# Neural network types
NN_TYPES = ["mlp", "cnn", "residual", "transformer"]
SELECTED_NN = "mlp"  # Default network type

# Create output directory structure
os.makedirs(OUTPUT_DIR, exist_ok=True)
for op in OPERATIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, op), exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable mixed precision if requested
if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

# Enable XLA if requested
if USE_XLA:
    tf.config.optimizer.set_jit(True)
    print("XLA optimization enabled")

# Initialize distributed strategy if requested
if USE_DISTRIBUTED:
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Distributed training enabled with {strategy.num_replicas_in_sync} devices")
    except:
        print("No multiple GPUs found, falling back to single device")
        strategy = tf.distribute.get_strategy()
else:
    strategy = tf.distribute.get_strategy()

# --- AES Constants and Tables ---

# AES S-box lookup table (SubBytes transformation)
AES_SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

# Inverse S-box for reference
AES_INV_SBOX = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
], dtype=np.uint8)

# MixColumns constants
MIX_COL_MATRIX = np.array([
    [0x02, 0x03, 0x01, 0x01],
    [0x01, 0x02, 0x03, 0x01],
    [0x01, 0x01, 0x02, 0x03],
    [0x03, 0x01, 0x01, 0x02]
], dtype=np.uint8)

# AES round constants
RCON = np.array([
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
], dtype=np.uint8)

# --- Core AES Operation Implementations ---

def bytes_to_state(data):
    """Convert 16 bytes to 4x4 AES state matrix"""
    state = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            state[j, i] = data[i * 4 + j]
    return state

def state_to_bytes(state):
    """Convert 4x4 AES state matrix to 16 bytes"""
    data = bytearray(16)
    for i in range(4):
        for j in range(4):
            data[i * 4 + j] = state[j, i]
    return bytes(data)

def sub_bytes(state):
    """Apply SubBytes (S-box) transformation to state"""
    for i in range(4):
        for j in range(4):
            state[i, j] = AES_SBOX[state[i, j]]
    return state

def shift_rows(state):
    """Apply ShiftRows transformation to state"""
    state[1] = np.roll(state[1], -1)    # Shift row 1 left by 1
    state[2] = np.roll(state[2], -2)    # Shift row 2 left by 2
    state[3] = np.roll(state[3], -3)    # Shift row 3 left by 3
    return state

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

def mix_columns(state):
    """Apply MixColumns transformation to state"""
    result = np.zeros_like(state)
    for col in range(4):
        for row in range(4):
            value = 0
            for i in range(4):
                value ^= galois_multiply(state[i, col], MIX_COL_MATRIX[row, i])
            result[row, col] = value
    return result

def add_round_key(state, round_key):
    """Apply AddRoundKey transformation to state"""
    key_state = bytes_to_state(round_key)
    for i in range(4):
        for j in range(4):
            state[i, j] ^= key_state[i, j]
    return state

def aes_full_round(state, round_key):
    """Perform one full round of AES"""
    state = sub_bytes(state)
    state = shift_rows(state)
    state = mix_columns(state)
    state = add_round_key(state, round_key)
    return state

def aes_key_schedule_core(word, iteration):
    """Core key schedule transformation"""
    # Rotate word
    word = np.roll(word, -1)
    
    # Apply S-box
    for i in range(4):
        word[i] = AES_SBOX[word[i]]
    
    # XOR with round constant
    word[0] ^= RCON[iteration]
    
    return word

def aes_key_expansion(key, rounds=10):
    """Expand AES key into round keys"""
    key_bytes = np.frombuffer(key, dtype=np.uint8)
    expanded_key = np.zeros((4 * (rounds + 1), 4), dtype=np.uint8)
    
    # First round key is the original key
    for i in range(4):
        expanded_key[i] = key_bytes[i*4:i*4+4]
    
    # Generate the rest of the round keys
    for i in range(4, 4 * (rounds + 1)):
        temp = expanded_key[i-1].copy()
        
        if i % 4 == 0:
            temp = aes_key_schedule_core(temp, i // 4 - 1)
        
        expanded_key[i] = expanded_key[i-4] ^ temp
    
    # Convert to round keys format
    round_keys = []
    for r in range(rounds + 1):
        round_key = bytes(expanded_key[r*4:(r+1)*4].flatten())
        round_keys.append(round_key)
    
    return round_keys

def aes_ablated_round(state, round_key, skip_operations):
    """Perform a round of AES with specified operations ablated (skipped)"""
    if "sub_bytes" not in skip_operations:
        state = sub_bytes(state)
    if "shift_rows" not in skip_operations:
        state = shift_rows(state)
    if "mix_columns" not in skip_operations:
        state = mix_columns(state)
    if "add_round_key" not in skip_operations:
        state = add_round_key(state, round_key)
    return state

# --- Dataset Generation for Individual AES Operations ---

def generate_sbox_dataset(num_samples):
    """Generate dataset for learning the S-box transformation"""
    X = np.random.randint(0, 256, size=(num_samples, 1), dtype=np.uint8)
    y = np.zeros_like(X)
    
    for i in range(num_samples):
        y[i, 0] = AES_SBOX[X[i, 0]]
    
    # Convert to one-hot encoding for inputs and outputs
    X_onehot = tf.keras.utils.to_categorical(X, num_classes=256)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=256)
    
    # Reshape to flatten the one-hot vectors
    X_onehot = X_onehot.reshape(num_samples, 256)
    y_onehot = y_onehot.reshape(num_samples, 256)
    
    return X_onehot.astype(np.float32), y_onehot.astype(np.float32), X, y

def generate_shiftrows_dataset(num_samples):
    """Generate dataset for learning the ShiftRows transformation"""
    X = np.zeros((num_samples, 16), dtype=np.uint8)
    y = np.zeros((num_samples, 16), dtype=np.uint8)
    
    for i in range(num_samples):
        # Generate random 16-byte block
        block = np.random.randint(0, 256, size=16, dtype=np.uint8)
        X[i] = block
        
        # Apply ShiftRows
        state = bytes_to_state(block)
        shifted_state = shift_rows(state.copy())
        
        # Fix: Use frombuffer instead of array for bytes conversion
        result_bytes = state_to_bytes(shifted_state)
        y[i] = np.frombuffer(result_bytes, dtype=np.uint8)
    
    # Convert to binary representation (more suitable for CNNs)
     # X_bin = np.unpackbits(X, axis=1)
     # y_bin = np.unpackbits(y, axis=1)
    
    # Ensure precise 0.0 and 1.0 values
     # X_bin = np.where(X_bin > 0, 1.0, 0.0)
     # y_bin = np.where(y_bin > 0, 1.0, 0.0)
    
    # Use direct byte values normalized to [0,1]
    X_norm = X.astype(np.float32) / 255.0
    y_norm = y.astype(np.float32) / 255.0
    
    # Return bits
     # return X_bin.astype(np.float32), y_bin.astype(np.float32), X, y
    
    # Return bytes
    return X_norm, y_norm, X, y

def generate_mixcolumns_dataset(num_samples):
    """Generate dataset for learning the MixColumns transformation"""
    X = np.zeros((num_samples, 16), dtype=np.uint8)
    y = np.zeros((num_samples, 16), dtype=np.uint8)
    
    for i in range(num_samples):
        # Generate random 16-byte block
        block = np.random.randint(0, 256, size=16, dtype=np.uint8)
        X[i] = block
        
        # Apply MixColumns
        state = bytes_to_state(block)
        mixed_state = mix_columns(state.copy())
        
        # Fix: Use frombuffer instead of array for bytes conversion
        result_bytes = state_to_bytes(mixed_state)
        y[i] = np.frombuffer(result_bytes, dtype=np.uint8)
    
    # Convert to binary representation
    X_bin = np.unpackbits(X, axis=1)
    y_bin = np.unpackbits(y, axis=1)
    
    # Ensure precise 0.0 and 1.0 values
    X_bin = np.where(X_bin > 0, 1.0, 0.0)
    y_bin = np.where(y_bin > 0, 1.0, 0.0)
    
    return X_bin.astype(np.float32), y_bin.astype(np.float32), X, y

def generate_addroundkey_dataset(num_samples, fixed_key=False):
    """Generate dataset for learning the AddRoundKey transformation"""
    X = np.zeros((num_samples, 32), dtype=np.uint8)  # 16 bytes data + 16 bytes key
    y = np.zeros((num_samples, 16), dtype=np.uint8)
    
    # Generate a fixed key if requested
    if fixed_key:
        key = np.random.randint(0, 256, size=16, dtype=np.uint8)
    
    for i in range(num_samples):
        # Generate random 16-byte block
        block = np.random.randint(0, 256, size=16, dtype=np.uint8)
        
        # Generate key or use fixed key
        if not fixed_key:
            key = np.random.randint(0, 256, size=16, dtype=np.uint8)
        
        # Store block and key
        X[i, :16] = block
        X[i, 16:] = key
        
        # Apply AddRoundKey
        state = bytes_to_state(block)
        keyed_state = add_round_key(state.copy(), key)
        y[i] = np.array(state_to_bytes(keyed_state), dtype=np.uint8)
    
    # Convert to binary representation
    X_bin = np.unpackbits(X, axis=1)
    y_bin = np.unpackbits(y, axis=1)
    
    return X_bin.astype(np.float32), y_bin.astype(np.float32), X, y

def generate_full_round_dataset(num_samples, fixed_key=False):
    """Generate dataset for learning a full AES round"""
    X = np.zeros((num_samples, 32), dtype=np.uint8)  # 16 bytes data + 16 bytes key
    y = np.zeros((num_samples, 16), dtype=np.uint8)
    
    # Generate a fixed key if requested
    if fixed_key:
        key = np.random.randint(0, 256, size=16, dtype=np.uint8)
    
    for i in range(num_samples):
        # Generate random 16-byte block
        block = np.random.randint(0, 256, size=16, dtype=np.uint8)
        
        # Generate key or use fixed key
        if not fixed_key:
            key = np.random.randint(0, 256, size=16, dtype=np.uint8)
        
        # Store block and key
        X[i, :16] = block
        X[i, 16:] = key
        
        # Apply full AES round
        state = bytes_to_state(block)
        round_state = aes_full_round(state.copy(), key)
        y[i] = np.array(state_to_bytes(round_state), dtype=np.uint8)
    
    # Convert to binary representation
    X_bin = np.unpackbits(X, axis=1)
    y_bin = np.unpackbits(y, axis=1)
    
    return X_bin.astype(np.float32), y_bin.astype(np.float32), X, y

def generate_ablated_round_dataset(num_samples, skip_operations, fixed_key=False):
    """Generate dataset for learning an AES round with certain operations ablated"""
    X = np.zeros((num_samples, 32), dtype=np.uint8)  # 16 bytes data + 16 bytes key
    y = np.zeros((num_samples, 16), dtype=np.uint8)
    
    # Generate a fixed key if requested
    if fixed_key:
        key = np.random.randint(0, 256, size=16, dtype=np.uint8)
    
    for i in range(num_samples):
        # Generate random 16-byte block
        block = np.random.randint(0, 256, size=16, dtype=np.uint8)
        
        # Generate key or use fixed key
        if not fixed_key:
            key = np.random.randint(0, 256, size=16, dtype=np.uint8)
        
        # Store block and key
        X[i, :16] = block
        X[i, 16:] = key
        
        # Apply ablated AES round
        state = bytes_to_state(block)
        round_state = aes_ablated_round(state.copy(), key, skip_operations)
        y[i] = np.array(state_to_bytes(round_state), dtype=np.uint8)
    
    # Convert to binary representation
    X_bin = np.unpackbits(X, axis=1)
    y_bin = np.unpackbits(y, axis=1)
    
    return X_bin.astype(np.float32), y_bin.astype(np.float32), X, y

def generate_key_schedule_dataset(num_samples, round_idx=1):
    """Generate dataset for learning the AES key schedule"""
    X = np.zeros((num_samples, 16), dtype=np.uint8)  # Input key
    y = np.zeros((num_samples, 16), dtype=np.uint8)  # Round key
    
    for i in range(num_samples):
        # Generate random 16-byte key
        key = bytes(np.random.randint(0, 256, size=16, dtype=np.uint8))
        X[i] = np.frombuffer(key, dtype=np.uint8)
        
        # Generate round keys
        round_keys = aes_key_expansion(key)
        y[i] = np.frombuffer(round_keys[round_idx], dtype=np.uint8)
    
    # Convert to binary representation
    X_bin = np.unpackbits(X, axis=1)
    y_bin = np.unpackbits(y, axis=1)
    
    return X_bin.astype(np.float32), y_bin.astype(np.float32), X, y

def generate_dataset(operation, num_samples, **kwargs):
    """Generate dataset for the specified AES operation"""
    print(f"Generating dataset for {operation} operation...")
    
    if operation == "sbox":
        X, y, X_raw, y_raw = generate_sbox_dataset(num_samples)
    elif operation == "shiftrows":
        X, y, X_raw, y_raw = generate_shiftrows_dataset(num_samples)
    elif operation == "mixcolumns":
        X, y, X_raw, y_raw = generate_mixcolumns_dataset(num_samples)
    elif operation == "addroundkey":
        X, y, X_raw, y_raw = generate_addroundkey_dataset(num_samples, **kwargs)
    elif operation == "full_round":
        X, y, X_raw, y_raw = generate_full_round_dataset(num_samples, **kwargs)
    elif operation == "key_schedule":
        X, y, X_raw, y_raw = generate_key_schedule_dataset(num_samples, **kwargs.get('round_idx', 1))
    elif operation == "ablated_round":
        # Special case for ablation studies
        skip_operations = kwargs.get('skip_operations', [])
        X, y, X_raw, y_raw = generate_ablated_round_dataset(num_samples, skip_operations, kwargs.get('fixed_key', False))
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    print(f"Generated {num_samples} samples with shapes: X={X.shape}, y={y.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_raw': X_raw,
        'y_raw': y_raw
    }

# --- Neural Network Architectures ---

def create_mlp_model(input_shape, output_shape):
    """Create a standard MLP model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(output_shape, activation='sigmoid')
    ])
    return model

def create_cnn_model(input_shape, output_shape):
    """Create a CNN model (better for spatial operations like ShiftRows)"""
    # Reshape the input to add spatial dimensions if necessary
    if len(input_shape) == 1:
        # For S-box, reshape to 16x16 grid (256 values)
        if input_shape[0] == 256:
            input_layer = layers.Input(shape=input_shape)
            x = layers.Reshape((16, 16, 1))(input_layer)
        # For other operations with binary representation
        else:
            # Reshape to make the bits into a 2D grid
            dim = int(np.sqrt(input_shape[0]))
            if dim * dim == input_shape[0]:
                input_layer = layers.Input(shape=input_shape)
                x = layers.Reshape((dim, dim, 1))(input_layer)
            else:
                # Fall back to 1D CNN if not a perfect square
                input_layer = layers.Input(shape=input_shape)
                x = layers.Reshape((input_shape[0], 1))(input_layer)
                x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
                x = layers.MaxPooling1D(pool_size=2)(x)
                x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
                x = layers.MaxPooling1D(pool_size=2)(x)
                x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
                x = layers.GlobalAveragePooling1D()(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(output_shape, activation='sigmoid')(x)
                return keras.Model(inputs=input_layer, outputs=x)
    else:
        input_layer = layers.Input(shape=input_shape)
        x = input_layer
    
    # Main CNN body (for 2D inputs)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(output_shape, activation='sigmoid')(x)
    
    return keras.Model(inputs=input_layer, outputs=x)

def create_residual_block(x, filters, kernel_size=3):
    """Create a residual block"""
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # If the shortcut needs to be adjusted to match dimensions
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1)(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_residual_model(input_shape, output_shape):
    """Create a residual neural network model"""
    # Handle reshaping if needed
    if len(input_shape) == 1:
        # For S-box, reshape to 16x16 grid (256 values)
        if input_shape[0] == 256:
            input_layer = layers.Input(shape=input_shape)
            x = layers.Reshape((16, 16, 1))(input_layer)
        # For other operations with binary representation
        else:
            # Try to reshape to a square if possible
            dim = int(np.sqrt(input_shape[0]))
            if dim * dim == input_shape[0]:
                input_layer = layers.Input(shape=input_shape)
                x = layers.Reshape((dim, dim, 1))(input_layer)
            else:
                # For non-square inputs, use a 1D approach
                input_layer = layers.Input(shape=input_shape)
                x = layers.Reshape((input_shape[0], 1))(input_layer)
                
                # 1D residual network
                x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                
                # 1D residual blocks
                for _ in range(3):
                    skip = x
                    x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Activation('relu')(x)
                    x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Add()([x, skip])
                    x = layers.Activation('relu')(x)
                
                x = layers.GlobalAveragePooling1D()(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.Dense(output_shape, activation='sigmoid')(x)
                
                return keras.Model(inputs=input_layer, outputs=x)
    else:
        input_layer = layers.Input(shape=input_shape)
        x = input_layer
    
    # Initial convolution
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = create_residual_block(x, 64)
    x = create_residual_block(x, 64)
    
    x = create_residual_block(x, 128)
    x = create_residual_block(x, 128)
    
    x = create_residual_block(x, 256)
    x = create_residual_block(x, 256)
    
    # Global pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(output_shape, activation='sigmoid')(x)
    
    return keras.Model(inputs=input_layer, outputs=x)

def get_positional_encoding(seq_len, d_model):
    """Generate positional encoding for transformer"""
    positions = np.arange(seq_len)[:, np.newaxis]
    depths = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / (10000**(depths / d_model))
    angle_rads = positions * angle_rates
    
    pos_encoding = np.zeros((1, seq_len, d_model))
    pos_encoding[0, :, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[0, :, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_transformer_model(input_shape, output_shape):
    """Create a transformer model"""
    input_layer = layers.Input(shape=input_shape)
    
    # Reshape for transformer if 1D input
    if len(input_shape) == 1:
        # Reshape to sequence
        seq_length = min(32, input_shape[0] // 8)  # Sequence length, with each step having 8 features
        feature_dim = input_shape[0] // seq_length
        
        x = layers.Reshape((seq_length, feature_dim))(input_layer)
        
        # Positional encoding
        positional_encoding = get_positional_encoding(seq_length, feature_dim)
        x = x + positional_encoding
    else:
        # For already 2D inputs
        seq_length = input_shape[0]
        feature_dim = input_shape[1]
        x = input_layer
    
    # Multi-head attention blocks
    for _ in range(6):
        # Self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=feature_dim // 8)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = layers.Dense(feature_dim * 4, activation='relu')(x)
        ffn = layers.Dense(feature_dim)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(output_shape, activation='sigmoid')(x)
    
    return keras.Model(inputs=input_layer, outputs=x)

def create_model(model_type, input_shape, output_shape):
    """Create a model based on the selected type"""
    if model_type == "mlp":
        return create_mlp_model(input_shape, output_shape)
    elif model_type == "cnn":
        return create_cnn_model(input_shape, output_shape)
    elif model_type == "residual":
        return create_residual_model(input_shape, output_shape)
    elif model_type == "transformer":
        return create_transformer_model(input_shape, output_shape)
    else:
        print(f"Unknown model type: {model_type}, falling back to MLP")
        return create_mlp_model(input_shape, output_shape)

# --- Model Training and Evaluation ---

def train_and_evaluate(dataset, operation, model_type=SELECTED_NN):
    """Train and evaluate a model for the specified operation"""
    print(f"\n{'-'*60}")
    print(f"Training {model_type.upper()} model for {operation} operation")
    print(f"{'-'*60}")
    
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_val, y_val = dataset['X_val'], dataset['y_val']
    X_test, y_test = dataset['X_test'], dataset['y_test']
    
    # Setup MLflow experiment
    mlflow.set_experiment(f"aes_{operation}_{model_type}")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("operation", operation)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("num_samples", len(X_train) + len(X_val) + len(X_test))
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        
        # Create model using distributed strategy
        with strategy.scope():
            # Create model
            model = create_model(model_type, X_train.shape[1:], y_train.shape[1])
            
            # Define optimizer with reduced learning rate
            optimizer = keras.optimizers.Adam(
                    learning_rate=0.0001,
                    clipnorm=1
                    )
            
            # Apply loss scaling if using mixed precision
            if USE_MIXED_PRECISION:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
            # Define loss based on the operation
            if operation == "sbox" and y_train.shape[1] == 256:
                # For S-box with one-hot encoding, use categorical crossentropy
                loss = 'categorical_crossentropy'
            else:
                # For binary output, use binary crossentropy
                loss = 'binary_crossentropy'
            
            # Compile model with the prepared optimizer
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy']
            )
        
        # Model summary
        model.summary()
        
        # Callbacks
        callback_list = []
        if USE_EARLY_STOPPING:
            callback_list.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=PATIENCE,
                    restore_best_weights=True
                )
            )
        
        callback_list.append(
            keras.callbacks.TerminateOnNaN()
        )       
        # TensorBoard logging
        log_dir = os.path.join(OUTPUT_DIR, operation, model_type, f"logs_{int(time.time())}")
        callback_list.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        )
        
        # Model checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, operation, model_type, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        callback_list.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_path, "model_ep{epoch:03d}_vl{val_loss:.4f}.keras"),
                save_best_only=True,
                monitor='val_loss'
            )
        )
        
        # MLflow callback for auto-logging
        callback_list.append(
            mlflow.tensorflow.MlflowCallback()
        )
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callback_list,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Print training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        print(f"Training completed in {hours}h {minutes}m {seconds}s")
        
        # Evaluate model
        print("\nEvaluating model on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("training_time", training_time)
        
        # Calculate per-bit accuracy if using binary representation
        if operation != "sbox" or y_train.shape[1] != 256:
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            bit_accuracies = []
            for i in range(y_test.shape[1]):
                bit_acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
                bit_accuracies.append(bit_acc)
            
            avg_bit_acc = np.mean(bit_accuracies)
            min_bit_acc = np.min(bit_accuracies)
            max_bit_acc = np.max(bit_accuracies)
            
            print(f"Average bit accuracy: {avg_bit_acc:.4f}")
            print(f"Minimum bit accuracy: {min_bit_acc:.4f}")
            print(f"Maximum bit accuracy: {max_bit_acc:.4f}")
            
            # Log bit accuracy metrics
            mlflow.log_metric("avg_bit_accuracy", avg_bit_acc)
            mlflow.log_metric("min_bit_accuracy", min_bit_acc)
            mlflow.log_metric("max_bit_accuracy", max_bit_acc)
            
            # Plot bit accuracies
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(bit_accuracies)), bit_accuracies)
            plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
            plt.xlabel('Bit Position')
            plt.ylabel('Accuracy')
            plt.title(f'Bit-level Accuracy for {operation.upper()} ({model_type.upper()} model)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            bit_acc_plot_path = os.path.join(OUTPUT_DIR, operation, f"{model_type}_bit_accuracies.png")
            plt.savefig(bit_acc_plot_path, dpi=300)
            mlflow.log_artifact(bit_acc_plot_path)
            plt.close()
        
        # Confusion matrix for S-box (if using one-hot encoding)
        if operation == "sbox" and y_train.shape[1] == 256:
            # Get predictions on test set
            X_raw_test = dataset['X_raw'][-len(X_test):]
            y_raw_test = dataset['y_raw'][-len(y_test):]
            
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Create confusion matrix
            cm = np.zeros((256, 256), dtype=np.int32)
            for i in range(len(X_raw_test)):
                x_val = X_raw_test[i, 0]
                true_y = y_raw_test[i, 0]
                pred_y = y_pred_classes[i]
                cm[x_val, pred_y] += 1
            
            # Visualize confusion matrix (sample of it)
            plt.figure(figsize=(10, 10))
            sample_size = 16  # Show a 16x16 sample of the confusion matrix
            sample_cm = cm[:sample_size, :sample_size]
            sns.heatmap(sample_cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'S-box Confusion Matrix Sample (First {sample_size}x{sample_size} entries)')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            cm_plot_path = os.path.join(OUTPUT_DIR, operation, f"{model_type}_confusion_matrix_sample.png")
            plt.savefig(cm_plot_path, dpi=300)
            mlflow.log_artifact(cm_plot_path)
            plt.close()
            
            # Calculate performance metrics for each value
            correct_predictions = np.zeros(256)
            total_occurrences = np.zeros(256)
            
            for i in range(256):
                total_occurrences[i] = np.sum(cm[i, :])
                correct_predictions[i] = cm[i, i]
            
            accuracy_per_value = correct_predictions / total_occurrences
            
            # Plot accuracy per value
            plt.figure(figsize=(12, 6))
            plt.bar(range(256), accuracy_per_value)
            plt.axhline(y=1/256, color='r', linestyle='--', label='Random guessing')
            plt.xlabel('Input Value')
            plt.ylabel('Accuracy')
            plt.title(f'S-box Accuracy per Input Value ({model_type.upper()} model)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            acc_plot_path = os.path.join(OUTPUT_DIR, operation, f"{model_type}_sbox_accuracy_per_value.png")
            plt.savefig(acc_plot_path, dpi=300)
            mlflow.log_artifact(acc_plot_path)
            plt.close()
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        history_plot_path = os.path.join(OUTPUT_DIR, operation, f"{model_type}_training_history.png")
        plt.savefig(history_plot_path, dpi=300)
        mlflow.log_artifact(history_plot_path)
        plt.close()
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, operation, f"{model_type}_model.keras")
        model.save(model_path)
        mlflow.tensorflow.log_model(model, "model")
        print(f"Model saved to {model_path}")
    
    # Return results
    return {
        'model': model,
        'history': history.history,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'training_time': training_time
    }

# --- Specialized Analysis Functions ---

def analyze_sbox_learning(model, dataset):
    """Analyze how well the model learned the S-box transformation"""
    print("\nAnalyzing S-box learning patterns...")
    
    # Generate all possible input values (0-255)
    all_inputs = np.arange(256).reshape(-1, 1)
    all_inputs_onehot = tf.keras.utils.to_categorical(all_inputs, num_classes=256)
    all_inputs_onehot = all_inputs_onehot.reshape(256, 256)
    
    # Get true S-box values
    true_outputs = np.array([AES_SBOX[i] for i in range(256)])
    
    # Get model predictions
    pred_outputs_onehot = model.predict(all_inputs_onehot)
    pred_outputs = np.argmax(pred_outputs_onehot, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(pred_outputs == true_outputs)
    print(f"Overall S-box accuracy: {accuracy:.4f}")
    
    # Find correctly and incorrectly predicted values
    correct_indices = np.where(pred_outputs == true_outputs)[0]
    incorrect_indices = np.where(pred_outputs != true_outputs)[0]
    
    print(f"Correctly predicted: {len(correct_indices)}/256 values")
    print(f"Incorrectly predicted: {len(incorrect_indices)}/256 values")
    
    # Analyze bit patterns in correctly vs incorrectly predicted values
    if len(correct_indices) > 0 and len(incorrect_indices) > 0:
        correct_inputs_bits = np.unpackbits(all_inputs[correct_indices].astype(np.uint8), axis=1)
        incorrect_inputs_bits = np.unpackbits(all_inputs[incorrect_indices].astype(np.uint8), axis=1)
        
        # Calculate average number of 1s in each set
        avg_bits_correct = np.mean(np.sum(correct_inputs_bits, axis=1))
        avg_bits_incorrect = np.mean(np.sum(incorrect_inputs_bits, axis=1))
        
        print(f"Average number of 1-bits in correctly predicted inputs: {avg_bits_correct:.2f}")
        print(f"Average number of 1-bits in incorrectly predicted inputs: {avg_bits_incorrect:.2f}")
    
    # Analyze prediction error distribution
    errors = np.abs(pred_outputs - true_outputs)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Average error magnitude: {avg_error:.2f}")
    print(f"Maximum error magnitude: {max_error}")
    
    # Plot error distribution
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=avg_error, color='r', linestyle='--', label=f'Average error: {avg_error:.2f}')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.title('S-box Prediction Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sbox', 'error_distribution.png'), dpi=300)
    plt.close()
    
    # Analyze cryptographic properties (nonlinearity, differential uniformity)
    analyze_sbox_cryptographic_properties(pred_outputs)
    
    return {
        'accuracy': accuracy,
        'correct_indices': correct_indices,
        'incorrect_indices': incorrect_indices,
        'errors': errors,
        'avg_error': avg_error,
        'max_error': max_error,
        'predicted_sbox': pred_outputs
    }

def analyze_sbox_cryptographic_properties(predicted_sbox):
    """Analyze cryptographic properties of the predicted S-box"""
    print("\nAnalyzing cryptographic properties of the predicted S-box...")
    
    # Analyze differential uniformity
    max_diff_count = 0
    diff_distribution = np.zeros((256, 256), dtype=np.int32)
    
    for input_diff in range(1, 256):
        for x in range(256):
            y = x ^ input_diff
            output_diff = predicted_sbox[x] ^ predicted_sbox[y]
            diff_distribution[input_diff, output_diff] += 1
            if diff_distribution[input_diff, output_diff] > max_diff_count:
                max_diff_count = diff_distribution[input_diff, output_diff]
    
    print(f"Differential uniformity: {max_diff_count} (AES S-box has 4)")
    
    # Compare with AES S-box differential uniformity
    aes_max_diff_count = 0
    aes_diff_distribution = np.zeros((256, 256), dtype=np.int32)
    
    for input_diff in range(1, 256):
        for x in range(256):
            y = x ^ input_diff
            output_diff = AES_SBOX[x] ^ AES_SBOX[y]
            aes_diff_distribution[input_diff, output_diff] += 1
            if aes_diff_distribution[input_diff, output_diff] > aes_max_diff_count:
                aes_max_diff_count = aes_diff_distribution[input_diff, output_diff]
    
    print(f"AES S-box differential uniformity: {aes_max_diff_count}")
    
    # Plot differential distribution comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(aes_diff_distribution[:16, :16], cmap='viridis', vmax=4)
    plt.title('AES S-box Differential Distribution (16x16 sample)')
    plt.xlabel('Output Difference')
    plt.ylabel('Input Difference')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(diff_distribution[:16, :16], cmap='viridis', vmax=4)
    plt.title('Predicted S-box Differential Distribution (16x16 sample)')
    plt.xlabel('Output Difference')
    plt.ylabel('Input Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sbox', 'differential_distribution.png'), dpi=300)
    plt.close()
    
    return {
        'diff_uniformity': max_diff_count,
        'aes_diff_uniformity': aes_max_diff_count,
        'diff_distribution': diff_distribution,
        'aes_diff_distribution': aes_diff_distribution
    }

def analyze_avalanche_effect(model, operation, dataset):
    """Analyze if the model exhibits the avalanche effect"""
    print("\nAnalyzing avalanche effect...")
    
    # Number of test cases
    num_tests = 1000
    
    if operation == "sbox":
        # For S-box, test single bit input changes
        inputs = np.random.randint(0, 256, size=(num_tests,), dtype=np.uint8)
        
        hamming_distances = []
        
        for i in range(num_tests):
            # Original input
            input_val = inputs[i]
            input_onehot = tf.keras.utils.to_categorical([input_val], num_classes=256)[0]
            
            # Get prediction for original input
            pred_orig = model.predict(input_onehot.reshape(1, 256))
            pred_orig_class = np.argmax(pred_orig)
            
            # Flip a random bit in the input
            bit_pos = np.random.randint(0, 8)
            mod_input_val = input_val ^ (1 << bit_pos)
            mod_input_onehot = tf.keras.utils.to_categorical([mod_input_val], num_classes=256)[0]
            
            # Get prediction for modified input
            pred_mod = model.predict(mod_input_onehot.reshape(1, 256))
            pred_mod_class = np.argmax(pred_mod)
            
            # Calculate Hamming distance between output bits
            orig_bits = np.unpackbits(np.array([pred_orig_class], dtype=np.uint8))
            mod_bits = np.unpackbits(np.array([pred_mod_class], dtype=np.uint8))
            hamming_dist = np.sum(orig_bits != mod_bits)
            hamming_distances.append(hamming_dist)
        
        # Calculate statistics
        avg_hamming_dist = np.mean(hamming_distances)
        
        # Calculate for real AES S-box for comparison
        aes_hamming_distances = []
        
        for i in range(num_tests):
            # Original input
            input_val = inputs[i]
            
            # Get AES S-box output
            orig_output = AES_SBOX[input_val]
            
            # Flip a random bit in the input
            bit_pos = np.random.randint(0, 8)
            mod_input_val = input_val ^ (1 << bit_pos)
            
            # Get AES S-box output for modified input
            mod_output = AES_SBOX[mod_input_val]
            
            # Calculate Hamming distance
            orig_bits = np.unpackbits(np.array([orig_output], dtype=np.uint8))
            mod_bits = np.unpackbits(np.array([mod_output], dtype=np.uint8))
            hamming_dist = np.sum(orig_bits != mod_bits)
            aes_hamming_distances.append(hamming_dist)
        
        avg_aes_hamming_dist = np.mean(aes_hamming_distances)
        
        print(f"Average Hamming distance for 1-bit input change (model): {avg_hamming_dist:.2f} bits")
        print(f"Average Hamming distance for 1-bit input change (AES S-box): {avg_aes_hamming_dist:.2f} bits")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.hist(hamming_distances, bins=range(9), alpha=0.7, label='Model')
        plt.hist(aes_hamming_distances, bins=range(9), alpha=0.7, label='AES S-box')
        plt.axvline(x=avg_hamming_dist, color='blue', linestyle='--', 
                    label=f'Model avg: {avg_hamming_dist:.2f} bits')
        plt.axvline(x=avg_aes_hamming_dist, color='orange', linestyle='--', 
                    label=f'AES avg: {avg_aes_hamming_dist:.2f} bits')
        plt.xlabel('Hamming Distance (bits)')
        plt.ylabel('Frequency')
        plt.title('Avalanche Effect: Hamming Distance for 1-bit Input Change')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, operation, 'avalanche_effect.png'), dpi=300)
        plt.close()
        
        return {
            'avg_hamming_dist': avg_hamming_dist,
            'avg_aes_hamming_dist': avg_aes_hamming_dist,
            'hamming_distances': hamming_distances,
            'aes_hamming_distances': aes_hamming_distances
        }
    
    elif operation in ["full_round", "ablated_round"]:
        # For full round or ablated round, test avalanche effect on the state
        X_test = dataset['X_test'][:num_tests]
        
        hamming_distances = []
        
        for i in range(num_tests):
            # Original input
            orig_input = X_test[i].copy()
            
            # Get prediction for original input
            pred_orig = model.predict(orig_input.reshape(1, -1))[0]
            pred_orig_binary = (pred_orig > 0.5).astype(int)
            
            # Flip a random bit in the input (only in the first 16 bytes, not in the key)
            bit_pos = np.random.randint(0, 128)  # First 16 bytes (128 bits)
            mod_input = orig_input.copy()
            mod_input[bit_pos] = 1 - mod_input[bit_pos]
            
            # Get prediction for modified input
            pred_mod = model.predict(mod_input.reshape(1, -1))[0]
            pred_mod_binary = (pred_mod > 0.5).astype(int)
            
            # Calculate Hamming distance between outputs
            hamming_dist = np.sum(pred_orig_binary != pred_mod_binary)
            hamming_distances.append(hamming_dist)
        
        # Calculate statistics
        avg_hamming_dist = np.mean(hamming_distances)
        print(f"Average Hamming distance for 1-bit input change: {avg_hamming_dist:.2f} bits")
        
        # Expected for a good avalanche effect: half the bits change
        expected_bits = 64  # Half of 128 bits
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(hamming_distances, bins=range(0, 129, 4), alpha=0.7)
        plt.axvline(x=avg_hamming_dist, color='r', linestyle='--', 
                    label=f'Average: {avg_hamming_dist:.2f} bits')
        plt.axvline(x=expected_bits, color='g', linestyle='--', 
                    label=f'Ideal avalanche: {expected_bits} bits')
        plt.xlabel('Hamming Distance (bits)')
        plt.ylabel('Frequency')
        plt.title('Avalanche Effect: Hamming Distance for 1-bit Input Change')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, operation, 'avalanche_effect.png'), dpi=300)
        plt.close()
        
        return {
            'avg_hamming_dist': avg_hamming_dist,
            'hamming_distances': hamming_distances,
            'ideal_hamming_dist': expected_bits
        }
    
    else:
        print(f"Avalanche effect analysis not implemented for {operation} operation")
        return None

def analyze_adversarial_examples(model, operation, dataset):
    """Generate adversarial examples to test the model's robustness"""
    print("\nGenerating adversarial examples...")
    
    # Number of examples to test
    num_examples = 100
    
    # Fast Gradient Sign Method parameters
    epsilon = 0.1  # Perturbation size
    
    # Select test samples
    X_test = dataset['X_test'][:num_examples]
    y_test = dataset['y_test'][:num_examples]
    
    # Convert to TensorFlow tensors
    X_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    adversarial_examples = []
    adversarial_success = []
    
    for i in range(num_examples):
        x = X_tensor[i:i+1]
        y_true = y_tensor[i:i+1]
        
        # Create adversarial example using FGSM
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = model(x)
            if operation == "sbox" and y_test.shape[1] == 256:
                loss = tf.keras.losses.categorical_crossentropy(y_true, prediction)
            else:
                loss = tf.keras.losses.binary_crossentropy(y_true, prediction)
        
        # Get the gradients
        gradient = tape.gradient(loss, x)
        
        # Create adversarial example
        signed_grad = tf.sign(gradient)
        adversarial_x = x + epsilon * signed_grad
        adversarial_x = tf.clip_by_value(adversarial_x, 0.0, 1.0)
        
        # Get predictions for original and adversarial examples
        original_pred = model.predict(x.numpy())
        adversarial_pred = model.predict(adversarial_x.numpy())
        
        # Check if attack was successful
        if operation == "sbox" and y_test.shape[1] == 256:
            original_class = np.argmax(original_pred[0])
            adversarial_class = np.argmax(adversarial_pred[0])
            success = (original_class != adversarial_class)
        else:
            original_binary = (original_pred > 0.5).astype(int)
            adversarial_binary = (adversarial_pred > 0.5).astype(int)
            hamming_dist = np.sum(original_binary != adversarial_binary)
            success = (hamming_dist > 0)
        
        adversarial_examples.append(adversarial_x.numpy()[0])
        adversarial_success.append(success)
    
    # Calculate success rate
    success_rate = np.mean(adversarial_success)
    print(f"Adversarial attack success rate: {success_rate:.4f}")
    
    # For binary output, calculate bit flip rate
    if operation != "sbox" or y_test.shape[1] != 256:
        bit_flip_counts = []
        for i in range(num_examples):
            original_pred = model.predict(X_test[i:i+1])
            adversarial_pred = model.predict(np.array([adversarial_examples[i]]))
            
            original_binary = (original_pred > 0.5).astype(int)
            adversarial_binary = (adversarial_pred > 0.5).astype(int)
            
            bit_flips = np.sum(original_binary != adversarial_binary)
            bit_flip_counts.append(bit_flips)
        
        avg_bit_flips = np.mean(bit_flip_counts)
        print(f"Average number of output bits flipped: {avg_bit_flips:.2f}")
        
        # Plot bit flip distribution
        plt.figure(figsize=(10, 6))
        plt.hist(bit_flip_counts, bins=range(0, max(bit_flip_counts) + 2), alpha=0.7)
        plt.axvline(x=avg_bit_flips, color='r', linestyle='--', 
                    label=f'Average: {avg_bit_flips:.2f} bits')
        plt.xlabel('Number of Output Bits Flipped')
        plt.ylabel('Frequency')
        plt.title('Effect of Adversarial Examples on Output Bits')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, operation, 'adversarial_bit_flips.png'), dpi=300)
        plt.close()
    
    return {
        'adversarial_examples': adversarial_examples,
        'success_rate': success_rate,
        'epsilon': epsilon
    }

def analyze_shiftrows_learning(model, dataset):
    """Analyze how well the model learned the ShiftRows transformation"""
    print("\nAnalyzing ShiftRows learning patterns...")
    
    # Generate test samples
    num_test_samples = 1000
    X_test = dataset['X_test'][:num_test_samples]
    y_test = dataset['y_test'][:num_test_samples]
    
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate bit-level accuracy
    bit_accuracies = []
    for i in range(y_test.shape[1]):
        bit_acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
        bit_accuracies.append(bit_acc)
    
    # Reshape bit accuracies to match byte structure
    byte_accuracies = np.zeros(16)
    for i in range(16):
        start_bit = i * 8
        end_bit = (i + 1) * 8
        byte_accuracies[i] = np.mean(bit_accuracies[start_bit:end_bit])
    
    # Reshape to 4x4 matrix to match AES state
    state_accuracies = bytes_to_state(byte_accuracies)
    
    print("Accuracy per byte position (4x4 state):")
    for row in state_accuracies:
        print(" ".join([f"{acc:.4f}" for acc in row]))
    
    # Visualize state accuracies
    plt.figure(figsize=(8, 6))
    sns.heatmap(state_accuracies, annot=True, fmt='.4f', cmap='Blues')
    plt.title('ShiftRows Accuracy per Byte Position')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shiftrows', 'state_accuracies.png'), dpi=300)
    plt.close()
    
    # Analyze row-wise accuracies
    row_accuracies = np.mean(state_accuracies, axis=1)
    print("\nAccuracy per row:")
    for i, acc in enumerate(row_accuracies):
        print(f"Row {i}: {acc:.4f}")
    
    # Plot row-wise accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(4), row_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Row Index')
    plt.ylabel('Accuracy')
    plt.title('ShiftRows Accuracy per Row')
    plt.xticks(range(4), [f'Row {i}' for i in range(4)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shiftrows', 'row_accuracies.png'), dpi=300)
    plt.close()
    
    return {
        'bit_accuracies': bit_accuracies,
        'byte_accuracies': byte_accuracies,
        'state_accuracies': state_accuracies,
        'row_accuracies': row_accuracies
    }

def analyze_mixcolumns_learning(model, dataset):
    """Analyze how well the model learned the MixColumns transformation"""
    print("\nAnalyzing MixColumns learning patterns...")
    
    # Generate test samples
    num_test_samples = 1000
    X_test = dataset['X_test'][:num_test_samples]
    y_test = dataset['y_test'][:num_test_samples]
    
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate bit-level accuracy
    bit_accuracies = []
    for i in range(y_test.shape[1]):
        bit_acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
        bit_accuracies.append(bit_acc)
    
    # Reshape bit accuracies to match byte structure
    byte_accuracies = np.zeros(16)
    for i in range(16):
        start_bit = i * 8
        end_bit = (i + 1) * 8
        byte_accuracies[i] = np.mean(bit_accuracies[start_bit:end_bit])
    
    # Reshape to 4x4 matrix to match AES state
    state_accuracies = bytes_to_state(byte_accuracies)
    
    print("Accuracy per byte position (4x4 state):")
    for row in state_accuracies:
        print(" ".join([f"{acc:.4f}" for acc in row]))
    
    # Visualize state accuracies
    plt.figure(figsize=(8, 6))
    sns.heatmap(state_accuracies, annot=True, fmt='.4f', cmap='Blues')
    plt.title('MixColumns Accuracy per Byte Position')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns', 'state_accuracies.png'), dpi=300)
    plt.close()
    
    # Analyze column-wise accuracies
    col_accuracies = np.mean(state_accuracies, axis=0)
    print("\nAccuracy per column:")
    for i, acc in enumerate(col_accuracies):
        print(f"Column {i}: {acc:.4f}")
    
    # Plot column-wise accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(4), col_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Column Index')
    plt.ylabel('Accuracy')
    plt.title('MixColumns Accuracy per Column')
    plt.xticks(range(4), [f'Column {i}' for i in range(4)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns', 'column_accuracies.png'), dpi=300)
    plt.close()
    
    # Analyze field operation learning
    analyze_galois_field_learning(model, dataset)
    
    return {
        'bit_accuracies': bit_accuracies,
        'byte_accuracies': byte_accuracies,
        'state_accuracies': state_accuracies,
        'col_accuracies': col_accuracies
    }

def analyze_galois_field_learning(model, dataset):
    """Analyze how well the model learned Galois field multiplication"""
    print("\nAnalyzing Galois field multiplication learning...")
    
    # Generate test cases for specific multiplications used in MixColumns
    multipliers = [0x01, 0x02, 0x03]  # The multipliers used in MixColumns
    
    # Generate all possible byte values
    values = np.arange(256, dtype=np.uint8)
    
    # Dictionary to store results for each multiplier
    results = {}
    
    for mult in multipliers:
        # Calculate expected results using Galois field multiplication
        expected_results = np.array([galois_multiply(mult, val) for val in values], dtype=np.uint8)
        
        # Create input dataset
        X = np.zeros((256, 16), dtype=np.uint8)
        for i in range(256):
            # Put the value in the first position
            X[i, 0] = values[i]
        
        # Convert to binary representation
        X_bin = np.unpackbits(X, axis=1).astype(np.float32)
        
        # Get model predictions
        y_pred = model.predict(X_bin)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Extract first byte predictions
        pred_bytes = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            pred_byte_bits = y_pred_binary[i, :8]  # First byte
            pred_bytes[i] = np.packbits(pred_byte_bits)[0]
        
        # Calculate accuracy
        accuracy = np.mean(pred_bytes == expected_results)
        results[mult] = {
            'accuracy': accuracy,
            'expected': expected_results,
            'predicted': pred_bytes
        }
        
        print(f"Accuracy for x{mult:02x} multiplication: {accuracy:.4f}")
    
    # Plot comparison for multiplier 0x02 (most important for diffusion)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(values, results[0x02]['expected'], label='Expected')
    plt.plot(values, results[0x02]['predicted'], 'o', alpha=0.5, label='Predicted')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title('Galois Field Multiplication by 0x02')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot errors
    errors = np.abs(results[0x02]['predicted'].astype(np.int32) - results[0x02]['expected'].astype(np.int32))
    
    plt.subplot(1, 2, 2)
    plt.stem(values, errors)
    plt.xlabel('Input Value')
    plt.ylabel('Error Magnitude')
    plt.title('Error in Galois Field Multiplication by 0x02')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns', 'galois_field_analysis.png'), dpi=300)
    plt.close()
    
    return results

def analyze_addroundkey_learning(model, dataset):
    """Analyze how well the model learned the AddRoundKey transformation"""
    print("\nAnalyzing AddRoundKey learning patterns...")
    
    # Generate test samples
    num_test_samples = 1000
    X_test = dataset['X_test'][:num_test_samples]
    y_test = dataset['y_test'][:num_test_samples]
    
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate bit-level accuracy
    bit_accuracies = []
    for i in range(y_test.shape[1]):
        bit_acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
        bit_accuracies.append(bit_acc)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(bit_accuracies)
    print(f"Overall bit accuracy: {overall_accuracy:.4f}")
    
    # Since AddRoundKey is a simple XOR, check if the model has learned this operation correctly
    # by testing some hand-crafted examples where we know the expected output
    
    # Generate some test plaintext blocks
    test_blocks = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    # Generate some test keys
    test_keys = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    
    # Prepare input for the model
    test_inputs = np.zeros((10, 32), dtype=np.uint8)
    test_inputs[:, :16] = test_blocks
    test_inputs[:, 16:] = test_keys
    
    # Calculate expected output
    expected_outputs = np.zeros((10, 16), dtype=np.uint8)
    for i in range(10):
        block_state = bytes_to_state(test_blocks[i])
        result_state = add_round_key(block_state.copy(), test_keys[i])
        expected_outputs[i] = np.array(state_to_bytes(result_state), dtype=np.uint8)
    
    # Convert to binary representation
    test_inputs_bin = np.unpackbits(test_inputs, axis=1)
    expected_outputs_bin = np.unpackbits(expected_outputs, axis=1)
    
    # Get model predictions
    test_preds = model.predict(test_inputs_bin)
    test_preds_bin = (test_preds > 0.5).astype(int)
    
    # Convert predictions back to bytes for comparison
    test_preds_bytes = np.zeros((10, 16), dtype=np.uint8)
    for i in range(10):
        # Reshape and pack bits
        bits = test_preds_bin[i].reshape(-1, 8)
        for j in range(16):
            test_preds_bytes[i, j] = np.packbits(bits[j])[0]
    
    # Calculate accuracy on the test examples
    test_accuracy = np.mean(test_preds_bytes == expected_outputs)
    print(f"XOR operation accuracy: {test_accuracy:.4f}")
    
    # Visualize the results for one example
    example_idx = 0
    
    print(f"\nExample {example_idx + 1}:")
    print(f"Plaintext: {test_blocks[example_idx].tobytes().hex()}")
    print(f"Key: {test_keys[example_idx].tobytes().hex()}")
    print(f"Expected: {expected_outputs[example_idx].tobytes().hex()}")
    print(f"Predicted: {test_preds_bytes[example_idx].tobytes().hex()}")
    
    # Calculate bit-level differences
    bit_diff = test_preds_bin[example_idx] != expected_outputs_bin[example_idx]
    print(f"Bit-level differences: {np.sum(bit_diff)}/{len(bit_diff)}")
    
    # Analyze XOR learning for each bit position
    xor_accuracies = []
    
    for bit_pos in range(8):  # Testing one byte is sufficient for XOR
        # Generate test samples for this bit position
        num_bit_samples = 1000
        bit_inputs = np.random.randint(0, 2, size=(num_bit_samples, 2))
        expected_xor = bit_inputs[:, 0] ^ bit_inputs[:, 1]
        
        # Create input in the expected format
        X_bit = np.zeros((num_bit_samples, X_test.shape[1]))
        
        # Set the specific bit in the first byte of plaintext and key
        for i in range(num_bit_samples):
            X_bit[i, bit_pos] = bit_inputs[i, 0]  # Set bit in plaintext
            X_bit[i, 128 + bit_pos] = bit_inputs[i, 1]  # Set bit in key
        
        # Get predictions
        bit_preds = model.predict(X_bit)
        bit_preds_binary = (bit_preds > 0.5).astype(int)
        
        # Extract the XOR result bit
        xor_results = bit_preds_binary[:, bit_pos]
        
        # Calculate accuracy
        xor_acc = np.mean(xor_results == expected_xor)
        xor_accuracies.append(xor_acc)
        
        print(f"XOR accuracy for bit position {bit_pos}: {xor_acc:.4f}")
    
    # Plot XOR accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(8), xor_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Bit Position')
    plt.ylabel('XOR Accuracy')
    plt.title('AddRoundKey XOR Learning per Bit Position')
    plt.xticks(range(8), [f'Bit {i}' for i in range(8)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'addroundkey', 'xor_accuracies.png'), dpi=300)
    plt.close()
    
    # Test key sensitivity
    key_sensitivity = analyze_key_sensitivity(model, dataset)
    
    return {
        'bit_accuracies': bit_accuracies,
        'overall_accuracy': overall_accuracy,
        'test_accuracy': test_accuracy,
        'xor_accuracies': xor_accuracies,
        'key_sensitivity': key_sensitivity,
        'example_results': {
            'plaintexts': test_blocks,
            'keys': test_keys,
            'expected': expected_outputs,
            'predicted': test_preds_bytes
        }
    }

def analyze_key_sensitivity(model, dataset):
    """Test the sensitivity of the model to key changes"""
    print("\nAnalyzing key sensitivity...")
    
    # Number of test samples
    num_samples = 100
    
    # Get test samples
    X_test = dataset['X_test'][:num_samples]
    
    # Dictionary to store results
    results = {
        'hamming_distances': [],
        'bit_flips': []
    }
    
    for i in range(num_samples):
        # Get original input
        orig_input = X_test[i].copy()
        
        # Make a copy with a flipped bit in the key
        mod_input = orig_input.copy()
        key_bit_pos = np.random.randint(128, 256)  # Key bits
        mod_input[key_bit_pos] = 1 - mod_input[key_bit_pos]
        
        # Get predictions
        orig_pred = model.predict(orig_input.reshape(1, -1))[0]
        mod_pred = model.predict(mod_input.reshape(1, -1))[0]
        
        # Convert to binary
        orig_binary = (orig_pred > 0.5).astype(int)
        mod_binary = (mod_pred > 0.5).astype(int)
        
        # Calculate Hamming distance
        hamming_dist = np.sum(orig_binary != mod_binary)
        results['hamming_distances'].append(hamming_dist)
        
        # Track which bits flipped
        flipped_bits = np.where(orig_binary != mod_binary)[0]
        results['bit_flips'].extend(flipped_bits)
    
    # Calculate statistics
    avg_hamming_dist = np.mean(results['hamming_distances'])
    print(f"Average Hamming distance for 1-bit key change: {avg_hamming_dist:.2f} bits")
    
    # Plot Hamming distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['hamming_distances'], bins=range(0, 129, 4), alpha=0.7)
    plt.axvline(x=avg_hamming_dist, color='r', linestyle='--', 
                label=f'Average: {avg_hamming_dist:.2f} bits')
    plt.axvline(x=64, color='g', linestyle='--', 
                label=f'Ideal: 64 bits (50%)')
    plt.xlabel('Hamming Distance (bits)')
    plt.ylabel('Frequency')
    plt.title('Key Sensitivity: Output Bits Changed for 1-bit Key Change')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'addroundkey', 'key_sensitivity.png'), dpi=300)
    plt.close()
    
    # Analyze which output bits are most affected by key changes
    if len(results['bit_flips']) > 0:
        bit_flip_counts = np.zeros(128)  # 128 output bits
        for bit in results['bit_flips']:
            if bit < 128:  # Make sure the bit index is valid
                bit_flip_counts[bit] += 1
        
        # Normalize by number of samples
        bit_flip_probs = bit_flip_counts / num_samples
        
        # Plot bit flip probabilities
        plt.figure(figsize=(12, 6))
        plt.bar(range(128), bit_flip_probs)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Ideal (50%)')
        plt.xlabel('Output Bit Position')
        plt.ylabel('Probability of Bit Flip')
        plt.title('Key Sensitivity: Output Bit Flip Probabilities')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'addroundkey', 'key_sensitivity_bits.png'), dpi=300)
        plt.close()
    
    return results

def analyze_full_round_learning(model, dataset):
    """Analyze how well the model learned a full AES round"""
    print("\nAnalyzing Full Round learning patterns...")
    
    # Generate test samples
    num_test_samples = 1000
    X_test = dataset['X_test'][:num_test_samples]
    y_test = dataset['y_test'][:num_test_samples]
    
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate bit-level accuracy
    bit_accuracies = []
    for i in range(y_test.shape[1]):
        bit_acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
        bit_accuracies.append(bit_acc)
    
    # Reshape bit accuracies to match byte structure
    byte_accuracies = np.zeros(16)
    for i in range(16):
        start_bit = i * 8
        end_bit = (i + 1) * 8
        byte_accuracies[i] = np.mean(bit_accuracies[start_bit:end_bit])
    
    # Reshape to 4x4 matrix to match AES state
    state_accuracies = bytes_to_state(byte_accuracies)
    
    print("Accuracy per byte position (4x4 state):")
    for row in state_accuracies:
        print(" ".join([f"{acc:.4f}" for acc in row]))
    
    # Visualize state accuracies
    plt.figure(figsize=(8, 6))
    sns.heatmap(state_accuracies, annot=True, fmt='.4f', cmap='Blues')
    plt.title('Full Round Accuracy per Byte Position')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'full_round', 'state_accuracies.png'), dpi=300)
    plt.close()
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(bit_accuracies)
    print(f"Overall bit accuracy: {overall_accuracy:.4f}")
    
    # Generate some test plaintext blocks
    test_blocks = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    # Generate some test keys
    test_keys = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    
    # Prepare input for the model
    test_inputs = np.zeros((10, 32), dtype=np.uint8)
    test_inputs[:, :16] = test_blocks
    test_inputs[:, 16:] = test_keys
    
    # Calculate expected output
    expected_outputs = np.zeros((10, 16), dtype=np.uint8)
    for i in range(10):
        block_state = bytes_to_state(test_blocks[i])
        result_state = aes_full_round(block_state.copy(), test_keys[i])
        expected_outputs[i] = np.array(state_to_bytes(result_state), dtype=np.uint8)
    
    # Convert to binary representation
    test_inputs_bin = np.unpackbits(test_inputs, axis=1).astype(np.float32)
    expected_outputs_bin = np.unpackbits(expected_outputs, axis=1).astype(np.float32)
    
    # Get model predictions
    test_preds = model.predict(test_inputs_bin)
    test_preds_bin = (test_preds > 0.5).astype(int)
    
    # Calculate accuracy on test examples
    test_bit_accuracy = np.mean(test_preds_bin == expected_outputs_bin)
    print(f"Test bit accuracy: {test_bit_accuracy:.4f}")
    
    # Calculate Hamming distance between predictions and expected outputs
    hamming_distances = []
    for i in range(10):
        hamming_dist = np.sum(test_preds_bin[i] != expected_outputs_bin[i])
        hamming_distances.append(hamming_dist)
    
    avg_hamming_dist = np.mean(hamming_distances)
    print(f"Average Hamming distance: {avg_hamming_dist:.2f} bits")
    
    # Plot Hamming distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(hamming_distances, bins=range(0, 128, 4), alpha=0.7)
    plt.axvline(x=avg_hamming_dist, color='r', linestyle='--', 
                label=f'Average distance: {avg_hamming_dist:.2f} bits')
    plt.axvline(x=64, color='g', linestyle='--', 
                label='Random guessing: 64 bits')
    plt.xlabel('Hamming Distance (bits)')
    plt.ylabel('Frequency')
    plt.title('Full Round Prediction Hamming Distance Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'full_round', 'hamming_distance_distribution.png'), dpi=300)
    plt.close()
    
    # Analyze avalanche effect
    avalanche_results = analyze_avalanche_effect(model, "full_round", dataset)
    
    # Analyze adversarial examples
    adversarial_results = analyze_adversarial_examples(model, "full_round", dataset)
    
    return {
        'bit_accuracies': bit_accuracies,
        'byte_accuracies': byte_accuracies,
        'state_accuracies': state_accuracies,
        'overall_accuracy': overall_accuracy,
        'test_bit_accuracy': test_bit_accuracy,
        'hamming_distances': hamming_distances,
        'avg_hamming_dist': avg_hamming_dist,
        'avalanche_results': avalanche_results,
        'adversarial_results': adversarial_results
    }

def analyze_key_schedule_learning(model, dataset):
    """Analyze how well the model learned the AES key schedule"""
    print("\nAnalyzing Key Schedule learning patterns...")
    
    # Generate test samples
    num_test_samples = 1000
    X_test = dataset['X_test'][:num_test_samples]
    y_test = dataset['y_test'][:num_test_samples]
    
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate bit-level accuracy
    bit_accuracies = []
    for i in range(y_test.shape[1]):
        bit_acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
        bit_accuracies.append(bit_acc)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(bit_accuracies)
    print(f"Overall bit accuracy: {overall_accuracy:.4f}")
    
    # Reshape bit accuracies to match byte structure
    byte_accuracies = np.zeros(16)
    for i in range(16):
        start_bit = i * 8
        end_bit = (i + 1) * 8
        byte_accuracies[i] = np.mean(bit_accuracies[start_bit:end_bit])
    
    # Print byte-level accuracies
    print("\nAccuracy per byte position:")
    for i, acc in enumerate(byte_accuracies):
        print(f"Byte {i}: {acc:.4f}")
    
    # Plot byte-level accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(range(16), byte_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Byte Position')
    plt.ylabel('Accuracy')
    plt.title('Key Schedule Accuracy per Byte Position')
    plt.xticks(range(16), [f'Byte {i}' for i in range(16)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'key_schedule', 'byte_accuracies.png'), dpi=300)
    plt.close()
    
    # Analyze key schedule core operation
    # The core operation involves rotation, S-box lookup, and XOR with round constant
    
    # Generate test key samples
    num_test_keys = 1000
    test_keys = np.random.randint(0, 256, size=(num_test_keys, 16), dtype=np.uint8)
    
    # Extract last column of the key
    last_columns = test_keys[:, 12:16]
    
    # Apply key schedule core to last column
    expected_core_outputs = np.zeros_like(last_columns)
    for i in range(num_test_keys):
        last_col = last_columns[i]
        
        # Apply core operation
        # 1. Rotate
        rotated = np.roll(last_col, -1)
        
        # 2. SubBytes
        for j in range(4):
            rotated[j] = AES_SBOX[rotated[j]]
        
        # 3. XOR with round constant (assuming first round)
        rotated[0] ^= RCON[0]
        
        expected_core_outputs[i] = rotated
    
    # Convert to model input format
    test_keys_bin = np.unpackbits(test_keys, axis=1).astype(np.float32)
    
    # Get model predictions
    preds = model.predict(test_keys_bin)
    preds_binary = (preds > 0.5).astype(int)
    
    # Extract first 4 bytes of predictions (corresponds to first column of next round key)
    pred_first_cols = np.zeros((num_test_keys, 4), dtype=np.uint8)
    for i in range(num_test_keys):
        for j in range(4):
            start_bit = j * 8
            end_bit = (j + 1) * 8
            pred_first_cols[i, j] = np.packbits(preds_binary[i, start_bit:end_bit])[0]
    
    # Calculate expected first columns
    expected_first_cols = np.zeros_like(pred_first_cols)
    for i in range(num_test_keys):
        first_col_key = test_keys[i, :4]
        expected_first_cols[i] = first_col_key ^ expected_core_outputs[i]
    
    # Calculate accuracy
    first_col_accuracy = np.mean(pred_first_cols == expected_first_cols)
    print(f"First column accuracy: {first_col_accuracy:.4f}")
    
    # Byte-level accuracy for first column
    for j in range(4):
        byte_acc = np.mean(pred_first_cols[:, j] == expected_first_cols[:, j])
        print(f"First column byte {j} accuracy: {byte_acc:.4f}")
    
    # Analyze key schedule expansion pattern for all rounds
    round_accuracies = []
    for round_idx in range(1, 10):  # Analyze rounds 1-9
        # Generate expected round keys
        round_key_accuracies = []
        
        # Sample a few keys for detailed analysis
        num_sample_keys = 10
        sample_keys = test_keys[:num_sample_keys]
        
        for key_idx in range(num_sample_keys):
            key = bytes(sample_keys[key_idx])
            round_keys = aes_key_expansion(key)
            
            # Compare model predictions with expected round keys
            # This is a simplified analysis - we're just checking if the model
            # seems to be learning the pattern of key expansion across rounds
            key_bin = np.unpackbits(sample_keys[key_idx]).astype(np.float32)
            pred = model.predict(key_bin.reshape(1, -1))[0]
            pred_binary = (pred > 0.5).astype(int)
            
            # Extract predicted round key
            pred_round_key = np.zeros(16, dtype=np.uint8)
            for i in range(16):
                start_bit = i * 8
                end_bit = (i + 1) * 8
                pred_round_key[i] = np.packbits(pred_binary[start_bit:end_bit])[0]
            
            # Expected round key
            expected_round_key = np.frombuffer(round_keys[round_idx], dtype=np.uint8)
            
            # Calculate accuracy
            round_key_acc = np.mean(pred_round_key == expected_round_key)
            round_key_accuracies.append(round_key_acc)
        
        avg_round_acc = np.mean(round_key_accuracies)
        round_accuracies.append(avg_round_acc)
        print(f"Round {round_idx} key accuracy: {avg_round_acc:.4f}")
    
    # Plot round key accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 10), round_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Key Schedule Accuracy per Round')
    plt.xticks(range(1, 10), [f'Round {i}' for i in range(1, 10)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'key_schedule', 'round_accuracies.png'), dpi=300)
    plt.close()
    
    return {
        'bit_accuracies': bit_accuracies,
        'byte_accuracies': byte_accuracies,
        'overall_accuracy': overall_accuracy,
        'first_col_accuracy': first_col_accuracy,
        'round_accuracies': round_accuracies
    }

def ablation_study_full_round():
    """Compare performance when certain AES operations are skipped"""
    print("\n" + "="*80)
    print("ABLATION STUDY: Comparing AES Round with Operations Removed")
    print("="*80)
    
    # Define operations to ablate
    operations_to_ablate = [
        [],  # Full round (baseline)
        ["sub_bytes"],
        ["shift_rows"],
        ["mix_columns"],
        ["add_round_key"],
        ["sub_bytes", "shift_rows"],
        ["mix_columns", "add_round_key"],
        ["sub_bytes", "mix_columns"],
        ["shift_rows", "add_round_key"]
    ]
    
    # Store results
    results = {}
    
    for skip_operations in operations_to_ablate:
        # Create operation name
        if not skip_operations:
            operation_name = "full_round"
        else:
            operation_name = "ablated_" + "_".join(skip_operations)
        
        print(f"\nAnalyzing round with {', '.join(skip_operations) if skip_operations else 'no'} operations skipped")
        
        # Generate dataset
        dataset = generate_dataset("ablated_round", NUM_SAMPLES, skip_operations=skip_operations)
        
        # Train and evaluate model
        model_results = train_and_evaluate(dataset, operation_name, SELECTED_NN)
        
        # Store results
        results[operation_name] = {
            'skip_operations': skip_operations,
            'test_accuracy': model_results['test_accuracy'],
            'training_time': model_results['training_time']
        }
    
    # Compare results
    print("\nAblation Study Results:")
    print(f"{'Operation':<30} {'Test Accuracy':<15} {'Training Time':<15}")
    print("-"*60)
    
    operation_names = []
    accuracies = []
    
    for operation_name, result in results.items():
        test_acc = result['test_accuracy']
        training_time = result['training_time']
        
        # Format training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"
        
        operation_label = "Full Round" if operation_name == "full_round" else f"Skip {', '.join(result['skip_operations'])}"
        print(f"{operation_label:<30} {test_acc:<15.4f} {time_str:<15}")
        
        operation_names.append(operation_label)
        accuracies.append(test_acc)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(operation_names, accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('AES Round Configuration')
    plt.ylabel('Test Accuracy')
    plt.title('Ablation Study: Effect of Removing AES Operations')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_study.png'), dpi=300)
    plt.close()
    
    # Compute and display feature importance
    baseline_acc = results['full_round']['test_accuracy']
    importance = {}
    
    for operation_name, result in results.items():
        if operation_name == "full_round" or len(result['skip_operations']) != 1:
            continue
        
        # Single operation ablated
        operation = result['skip_operations'][0]
        ablated_acc = result['test_accuracy']
        
        # Calculate importance as drop in accuracy
        importance[operation] = baseline_acc - ablated_acc
    
    # Plot feature importance
    if importance:
        operations = list(importance.keys())
        importance_values = [importance[op] for op in operations]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(operations, importance_values)
        plt.xlabel('AES Operation')
        plt.ylabel('Importance (Decrease in Accuracy)')
        plt.title('AES Operation Importance')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'operation_importance.png'), dpi=300)
        plt.close()
    
    return results

def analyze_operation_learning(operation, model, dataset):
    """Analyze how well the model learned the specified operation"""
    if operation == "sbox":
        return analyze_sbox_learning(model, dataset)
    elif operation == "shiftrows":
        return analyze_shiftrows_learning(model, dataset)
    elif operation == "mixcolumns":
        return analyze_mixcolumns_learning(model, dataset)
    elif operation == "addroundkey":
        return analyze_addroundkey_learning(model, dataset)
    elif operation == "full_round":
        return analyze_full_round_learning(model, dataset)
    elif operation == "key_schedule":
        return analyze_key_schedule_learning(model, dataset)
    elif operation.startswith("ablated_"):
        return analyze_full_round_learning(model, dataset)  # Use same analysis as full round
    else:
        print(f"No specific analysis available for operation: {operation}")
        return None

def compare_models_for_operation(operation, nn_types=NN_TYPES):
    """Train and compare different model types for a specific operation"""
    print(f"\n{'='*80}")
    print(f"Comparing models for {operation.upper()} operation")
    print(f"{'='*80}")
    
    # Generate dataset
    dataset = generate_dataset(operation, NUM_SAMPLES)
    
    # Train and evaluate each model type
    results = {}
    for model_type in nn_types:
        print(f"\nTraining {model_type.upper()} model...")
        model_results = train_and_evaluate(dataset, operation, model_type)
        
        # Analyze learning patterns
        analysis_results = analyze_operation_learning(operation, model_results['model'], dataset)
        
        results[model_type] = {
            'training_results': model_results,
            'analysis_results': analysis_results
        }
    
    # Compare results
    print(f"\n{'-'*60}")
    print(f"Comparison of models for {operation.upper()} operation")
    print(f"{'-'*60}")
    
    print(f"{'Model Type':<12} {'Test Accuracy':<15} {'Training Time':<15}")
    print(f"{'-'*60}")
    
    for model_type, result in results.items():
        test_acc = result['training_results']['test_accuracy']
        training_time = result['training_results']['training_time']
        
        # Format training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"
        
        print(f"{model_type.upper():<12} {test_acc:<15.4f} {time_str:<15}")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[model]['training_results']['test_accuracy'] for model in model_names]
    
    plt.bar(model_names, accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing (binary)')
    
    if operation == "sbox":
        plt.axhline(y=1/256, color='g', linestyle='--', label='Random guessing (S-box)')
    
    plt.xlabel('Model Type')
    plt.ylabel('Test Accuracy')
    plt.title(f'Model Comparison for {operation.upper()} Operation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, operation, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, operation, 'model_comparison_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results

# -------------------------------------------------------------------------
# Enhanced Models with GF Integration
# -------------------------------------------------------------------------

def load_gf_models(gf_model_dir="galois_field_learning"):
    """Load the pre-trained Galois Field models"""
    print("Loading pre-trained Galois Field models...")
    
    # Paths to pre-trained models
    gf_mul2_path = os.path.join(gf_model_dir, "gf_mul2", "structured_model.h5")
    gf_mul3_path = os.path.join(gf_model_dir, "gf_mul3", "complex_model.h5")
    
    # Load models if they exist
    if os.path.exists(gf_mul2_path) and os.path.exists(gf_mul3_path):
        gf_mul2_model = keras.models.load_model(gf_mul2_path)
        gf_mul3_model = keras.models.load_model(gf_mul3_path)
        print("Pre-trained GF models loaded successfully")
        return gf_mul2_model, gf_mul3_model
    else:
        print("Pre-trained GF models not found. Please run the Galois Field training first.")
        return None, None

# Create a custom layer for GF(2^8) multiplication
class GFMultiplyLayer(layers.Layer):
    """Layer that applies a pre-trained GF(2^8) multiplication model"""
    def __init__(self, gf_model, **kwargs):
        super(GFMultiplyLayer, self).__init__(**kwargs)
        self.gf_model = gf_model
    
    def call(self, inputs):
        # Ensure input is properly shaped for the GF model
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, 1])  # Reshape to [batch_size*elements, 1]
        
        # Apply the GF model
        result = self.gf_model(flattened)
        
        # Reshape back to original shape
        return tf.reshape(result, input_shape)

# Custom XOR layer for binary operations
class BitXORLayer(layers.Layer):
    """Layer that approximates bitwise XOR operation"""
    def __init__(self, **kwargs):
        super(BitXORLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Assumes inputs is a list of two tensors with identical shapes
        x1, x2 = inputs
        
        # Scale to [0,1] range if not already
        x1 = tf.clip_by_value(x1, 0, 1)
        x2 = tf.clip_by_value(x2, 0, 1)
        
        # Approximate XOR: (x1 * (1 - x2)) + (x2 * (1 - x1))
        return (x1 * (1 - x2)) + (x2 * (1 - x1))

def create_enhanced_mixcolumns_model(gf_mul2_model, gf_mul3_model):
    """Create a model for MixColumns using pre-trained GF components"""
    inputs = layers.Input(shape=(16,))  # 16 bytes of state
    
    # Normalize inputs to [0,1] range (required for GF models)
    x = layers.Lambda(lambda x: x / 255.0)(inputs)
    
    # Reshape to 4x4 state
    state = layers.Reshape((4, 4))(x)
    
    # Create GF multiplication layers
    gf_mul2 = GFMultiplyLayer(gf_mul2_model)
    gf_mul3 = GFMultiplyLayer(gf_mul3_model)
    
    # Process each column
    output_columns = []
    for col in range(4):
        # Extract column
        col_state = layers.Lambda(lambda x: x[:, :, col])(state)
        
        # Split into individual bytes
        s0 = layers.Lambda(lambda x: x[:, 0])(col_state)
        s1 = layers.Lambda(lambda x: x[:, 1])(col_state)
        s2 = layers.Lambda(lambda x: x[:, 2])(col_state)
        s3 = layers.Lambda(lambda x: x[:, 3])(col_state)
        
        # Create expanded dimensions for operations
        s0_exp = layers.Reshape((-1, 1))(s0)
        s1_exp = layers.Reshape((-1, 1))(s1)
        s2_exp = layers.Reshape((-1, 1))(s2)
        s3_exp = layers.Reshape((-1, 1))(s3)
        
        # Apply MixColumns formula for each row using our pre-trained GF models
        # Row 0: (2*s0) ⊕ (3*s1) ⊕ s2 ⊕ s3
        mul2_s0 = gf_mul2(s0_exp)
        mul3_s1 = gf_mul3(s1_exp)
        
        # Reshape back to vectors
        mul2_s0 = layers.Reshape((-1,))(mul2_s0)
        mul3_s1 = layers.Reshape((-1,))(mul3_s1)
        
        # Combine with XOR
        row0 = BitXORLayer()([mul2_s0, mul3_s1])
        row0 = BitXORLayer()([row0, s2])
        row0 = BitXORLayer()([row0, s3])
        
        # Row 1: s0 ⊕ (2*s1) ⊕ (3*s2) ⊕ s3
        mul2_s1 = gf_mul2(s1_exp)
        mul3_s2 = gf_mul3(s2_exp)
        
        mul2_s1 = layers.Reshape((-1,))(mul2_s1)
        mul3_s2 = layers.Reshape((-1,))(mul3_s2)
        
        row1 = BitXORLayer()([s0, mul2_s1])
        row1 = BitXORLayer()([row1, mul3_s2])
        row1 = BitXORLayer()([row1, s3])
        
        # Row 2: s0 ⊕ s1 ⊕ (2*s2) ⊕ (3*s3)
        mul2_s2 = gf_mul2(s2_exp)
        mul3_s3 = gf_mul3(s3_exp)
        
        mul2_s2 = layers.Reshape((-1,))(mul2_s2)
        mul3_s3 = layers.Reshape((-1,))(mul3_s3)
        
        row2 = BitXORLayer()([s0, s1])
        row2 = BitXORLayer()([row2, mul2_s2])
        row2 = BitXORLayer()([row2, mul3_s3])
        
        # Row 3: (3*s0) ⊕ s1 ⊕ s2 ⊕ (2*s3)
        mul3_s0 = gf_mul3(s0_exp)
        mul2_s3 = gf_mul2(s3_exp)
        
        mul3_s0 = layers.Reshape((-1,))(mul3_s0)
        mul2_s3 = layers.Reshape((-1,))(mul2_s3)
        
        row3 = BitXORLayer()([mul3_s0, s1])
        row3 = BitXORLayer()([row3, s2])
        row3 = BitXORLayer()([row3, mul2_s3])
        
        # Reshape and stack the rows for this column
        col_output = tf.stack([row0, row1, row2, row3], axis=1)
        output_columns.append(col_output)
    
    # Stack columns to recreate the state
    output_state = tf.stack(output_columns, axis=2)
    
    # Reshape to match expected output
    output_flat = layers.Reshape((16,))(output_state)
    
    # Scale back to [0,255] range
    outputs = layers.Lambda(lambda x: x * 255.0)(output_flat)
    
    return keras.Model(inputs=inputs, outputs=outputs)


# -------------------------------------------------------------------------
# Enhanced MixColumns Training and Evaluation
# -------------------------------------------------------------------------

def train_enhanced_mixcolumns_model(dataset=None, num_samples=100000):
    """Train the enhanced MixColumns model with GF components"""
    print("\n" + "="*80)
    print("Training Enhanced MixColumns Model with Galois Field Components")
    print("="*80)
    
    # Load pre-trained GF models
    gf_mul2_model, gf_mul3_model = load_gf_models()
    
    if gf_mul2_model is None or gf_mul3_model is None:
        print("Cannot train enhanced model without pre-trained GF components")
        return None
    
    # Generate or use provided dataset
    if dataset is None:
        dataset = generate_mixcolumns_dataset(num_samples)
    
    # Create enhanced model
    enhanced_model = create_enhanced_mixcolumns_model(gf_mul2_model, gf_mul3_model)
    enhanced_model.summary()
    
    # Prepare data
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    # Custom callback to track byte accuracy during training
    class ByteAccuracyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Check every 5 epochs to save computation
                y_pred = self.model.predict(X_val)
                y_pred_bytes = np.round(y_pred).astype(np.uint8)
                y_true_bytes = y_val
                
                # Calculate byte accuracy
                byte_acc = np.mean(y_pred_bytes == y_true_bytes)
                print(f"\nEpoch {epoch}: Byte accuracy on validation set: {byte_acc:.4f}")
    
    # Compile the model
    enhanced_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train the model
    history = enhanced_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[
            ByteAccuracyCallback(),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_DIR, 'enhanced_mixcolumns', 'logs'))
        ]
    )
    
    # Evaluate the model
    test_loss = enhanced_model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")
    
    # Save the model
    model_dir = os.path.join(OUTPUT_DIR, 'enhanced_mixcolumns')
    os.makedirs(model_dir, exist_ok=True)
    enhanced_model.save(os.path.join(model_dir, 'enhanced_model.h5'))
    
    # Analyze the model
    analyze_enhanced_mixcolumns(enhanced_model, dataset)
    
    return enhanced_model, history

def analyze_enhanced_mixcolumns(model, dataset):
    """Analyze the enhanced MixColumns model performance"""
    print("\nAnalyzing Enhanced MixColumns model performance...")
    
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Round to nearest integer for byte comparison
    y_pred_bytes = np.round(y_pred).astype(np.uint8)
    
    # Calculate byte-level accuracy
    byte_accuracy = np.mean(y_pred_bytes == y_test)
    print(f"Byte-level accuracy: {byte_accuracy:.4f}")
    
    # Calculate how many state matrices are perfectly predicted
    num_test_samples = y_test.shape[0]
    perfect_states = 0
    
    for i in range(num_test_samples):
        if np.all(y_pred_bytes[i] == y_test[i]):
            perfect_states += 1
    
    print(f"Perfect state predictions: {perfect_states}/{num_test_samples} ({perfect_states/num_test_samples:.4f})")
    
    # Calculate accuracy for each column
    column_accuracies = []
    for col in range(4):
        col_indices = np.array([col*4 + i for i in range(4)])
        col_accuracy = np.mean(y_pred_bytes[:, col_indices] == y_test[:, col_indices])
        column_accuracies.append(col_accuracy)
        print(f"Column {col} accuracy: {col_accuracy:.4f}")
    
    # Calculate error distribution
    errors = np.abs(y_pred_bytes - y_test)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Mean error: {mean_error:.2f}")
    print(f"Max error: {max_error}")
    
    # Calculate error histogram
    error_counts = np.bincount(errors.flatten(), minlength=256)
    print(f"Error distribution: {error_counts[:10]}...")
    
    # Calculate bit-level accuracy
    y_pred_bits = np.unpackbits(y_pred_bytes.reshape(-1, 1), axis=1).reshape(y_test.shape[0], -1)
    y_test_bits = np.unpackbits(y_test.reshape(-1, 1), axis=1).reshape(y_test.shape[0], -1)
    
    bit_accuracy = np.mean(y_pred_bits == y_test_bits)
    print(f"Bit-level accuracy: {bit_accuracy:.4f}")
    
    # Visualize column accuracies
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(range(4), column_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.xlabel('Column Index')
    plt.ylabel('Accuracy')
    plt.title('Enhanced MixColumns - Column Accuracy')
    plt.xticks(range(4), [f'Column {i}' for i in range(4)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_dir = os.path.join(OUTPUT_DIR, 'enhanced_mixcolumns')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'column_accuracies.png'), dpi=300)
    plt.close()
    
    # Visualize error distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(min(20, len(error_counts))), error_counts[:20])
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.title('Enhanced MixColumns - Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    return {
        'byte_accuracy': byte_accuracy,
        'perfect_states': perfect_states / num_test_samples,
        'column_accuracies': column_accuracies,
        'bit_accuracy': bit_accuracy,
        'mean_error': mean_error,
        'max_error': max_error
    }

# -------------------------------------------------------------------------
# Main Function for Enhanced Testing
# -------------------------------------------------------------------------

def test_enhanced_mixcolumns():
    """Test the enhanced MixColumns approach and compare with standard models"""
    print("\n" + "="*80)
    print("Testing Enhanced MixColumns Approach with Galois Field Integration")
    print("="*80)
    
    # Generate a smaller dataset for testing
    dataset = generate_mixcolumns_dataset(50000)
    
    # First, train a standard model for comparison
    print("\nTraining standard MLP model for comparison...")
    standard_model = create_mlp_model((128,), 128)  # Using bit representation
    
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_val, y_val = dataset['X_val'], dataset['y_val'] 
    X_test, y_test = dataset['X_test'], dataset['y_test']
    
    # Convert to bit representation for standard model
    X_train_bits = np.unpackbits(X_train.reshape(-1, 1), axis=1).reshape(X_train.shape[0], -1)
    y_train_bits = np.unpackbits(y_train.reshape(-1, 1), axis=1).reshape(y_train.shape[0], -1)
    
    X_val_bits = np.unpackbits(X_val.reshape(-1, 1), axis=1).reshape(X_val.shape[0], -1)
    y_val_bits = np.unpackbits(y_val.reshape(-1, 1), axis=1).reshape(y_val.shape[0], -1)
    
    X_test_bits = np.unpackbits(X_test.reshape(-1, 1), axis=1).reshape(X_test.shape[0], -1)
    y_test_bits = np.unpackbits(y_test.reshape(-1, 1), axis=1).reshape(y_test.shape[0], -1)
    
    # Train standard model
    standard_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    standard_history = standard_model.fit(
        X_train_bits, y_train_bits,
        validation_data=(X_val_bits, y_val_bits),
        epochs=30,
        batch_size=256,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Evaluate standard model
    standard_loss, standard_acc = standard_model.evaluate(X_test_bits, y_test_bits)
    print(f"Standard model test accuracy: {standard_acc:.4f}")
    
    # Now train the enhanced model
    enhanced_model, enhanced_history = train_enhanced_mixcolumns_model(dataset=dataset)
    
    # Compare both models
    print("\n" + "="*80)
    print("Comparison of Standard vs Enhanced Models for MixColumns")
    print("="*80)
    
    # For standard model - calculate byte-level accuracy
    y_pred_std = standard_model.predict(X_test_bits)
    y_pred_std_binary = (y_pred_std > 0.5).astype(np.float32)
    
    # Reshape to bytes (8 bits per byte)
    y_pred_std_binary_reshaped = y_pred_std_binary.reshape(-1, 8)
    y_test_bits_reshaped = y_test_bits.reshape(-1, 8)
    
    # Check if all bits in a byte match
    byte_correct = 0
    total_bytes = y_pred_std_binary_reshaped.shape[0]
    
    for i in range(total_bytes):
        if np.array_equal(y_pred_std_binary_reshaped[i], y_test_bits_reshaped[i]):
            byte_correct += 1
    
    std_byte_accuracy = byte_correct / total_bytes
    print(f"Standard model byte-level accuracy: {std_byte_accuracy:.4f}")
    
    # For enhanced model - get the previously calculated accuracy
    enhanced_results = analyze_enhanced_mixcolumns(enhanced_model, dataset)
    print(f"Enhanced model byte-level accuracy: {enhanced_results['byte_accuracy']:.4f}")
    
    print("\nConclusions:")
    if enhanced_results['byte_accuracy'] > std_byte_accuracy:
        improvement = enhanced_results['byte_accuracy'] / max(std_byte_accuracy, 1e-6)
        print(f"The enhanced model performs {improvement:.2f}x better than the standard model")
        print("This demonstrates that incorporating cryptographic structure helps neural networks")
        print("learn operations that are otherwise unlearnable with standard architectures.")
    else:
        print("The standard model outperformed the enhanced model. This is unexpected and suggests")
        print("that either the GF components need improvement or the integration approach should be revised.")
    
    return {
        'standard_model': standard_model,
        'enhanced_model': enhanced_model,
        'standard_byte_accuracy': std_byte_accuracy,
        'enhanced_byte_accuracy': enhanced_results['byte_accuracy'],
        'improvement_factor': enhanced_results['byte_accuracy'] / max(std_byte_accuracy, 1e-6)
    }

# --- Main Execution ---
def main():
    """Main execution function"""
    print(f"\n{'='*80}")
    print("AES Piecewise Analysis Framework")
    print(f"{'='*80}")
    
    # Check if operations are specified as a single operation or a list
    if isinstance(SELECTED_OPERATION, str) and SELECTED_OPERATION in OPERATIONS:
        selected_operations = [SELECTED_OPERATION]
    else:
        selected_operations = OPERATIONS
    
    if len(selected_operations) == 1:
        # Train and evaluate for a single operation
        operation = selected_operations[0]
        print(f"Analyzing {operation.upper()} operation")
        
        # Generate dataset
        dataset = generate_dataset(operation, NUM_SAMPLES)
        
        # Train and evaluate model
        results = train_and_evaluate(dataset, operation, SELECTED_NN)
        
        # Analyze learning patterns
        analysis_results = analyze_operation_learning(operation, results['model'], dataset)
        
    else:
        # Compare different operations
        print("Comparing different AES operations")
        
        # Train and compare models for each operation
        op_results = {}
        for operation in selected_operations:
            print(f"\nAnalyzing {operation.upper()} operation")
            op_results[operation] = compare_models_for_operation(operation, [SELECTED_NN])
        
        # Compare learnability across operations
        print(f"\n{'-'*60}")
        print("Learnability comparison across operations")
        print(f"{'-'*60}")
        
        print(f"{'Operation':<12} {'Test Accuracy':<15}")
        print(f"{'-'*60}")
        
        accuracies = []
        op_names = []
        
        for operation, result in op_results.items():
            model_result = result[SELECTED_NN]['training_results']
            test_acc = model_result['test_accuracy']
            
            print(f"{operation.upper():<12} {test_acc:<15.4f}")
            
            accuracies.append(test_acc)
            op_names.append(operation.upper())
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        plt.bar(op_names, accuracies)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing (binary)')
        plt.axhline(y=1/256, color='g', linestyle='--', label='Random guessing (S-box)')
        plt.xlabel('AES Operation')
        plt.ylabel('Test Accuracy')
        plt.title(f'Learnability Comparison Across AES Operations ({SELECTED_NN.upper()} model)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'operation_comparison.png'), dpi=300)
        plt.close()
    
    # Run ablation study if requested
    if "ablated_round" in selected_operations or "full_round" in selected_operations:
        ablation_results = ablation_study_full_round()
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
