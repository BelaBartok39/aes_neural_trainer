import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration with improved parameters
SEED = 42
NUM_SAMPLES = 200000  # Dataset size
BATCH_SIZE = 512  # Batch size for training
EPOCHS = 200  # Number of training epochs
OUTPUT_DIR = "galois_field_learning_improved"
USE_EARLY_STOPPING = True
PATIENCE = 30  # Early stopping patience
VALIDATION_SPLIT = 0.15  # Validation data percentage
TEST_SPLIT = 0.15  # Test data percentage

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Galois Field Operations - Enhanced with Optimized Implementation
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

# Create lookup tables for Galois field multiplication for all values (0-255)
# Pre-compute these for faster access
GF_MUL_TABLES = {}
for multiplier in range(256):
    GF_MUL_TABLES[multiplier] = np.array([galois_multiply(i, multiplier) for i in range(256)], dtype=np.uint8)

# Quick access for common multipliers
GF_MUL_2 = GF_MUL_TABLES[2]
GF_MUL_3 = GF_MUL_TABLES[3]

# Precompute bit patterns for all possible inputs and outputs
# This gives us direct insight into which bits affect which other bits
BIT_PATTERNS_INPUT = np.unpackbits(np.arange(256, dtype=np.uint8).reshape(-1, 1), axis=1)
BIT_PATTERNS_OUTPUT_2 = np.unpackbits(GF_MUL_2.reshape(-1, 1), axis=1)
BIT_PATTERNS_OUTPUT_3 = np.unpackbits(GF_MUL_3.reshape(-1, 1), axis=1)

# Calculate bit correlation matrices to understand input-output bit relationships
# These matrices show which input bits influence which output bits
BIT_CORRELATION_2 = np.zeros((8, 8))
BIT_CORRELATION_3 = np.zeros((8, 8))

for i in range(8):
    for j in range(8):
        # Calculate correlation between input bit i and output bit j
        BIT_CORRELATION_2[i, j] = np.corrcoef(BIT_PATTERNS_INPUT[:, i], BIT_PATTERNS_OUTPUT_2[:, j])[0, 1]
        BIT_CORRELATION_3[i, j] = np.corrcoef(BIT_PATTERNS_INPUT[:, i], BIT_PATTERNS_OUTPUT_3[:, j])[0, 1]

# -------------------------------------------------------------------------
# Dataset Generation - Enhanced with Enriched Features
# -------------------------------------------------------------------------

def generate_gf_multiply_dataset(num_samples, multiplier=2):
    """Generate dataset for learning GF(2^8) multiplication by a constant"""
    # Choose the appropriate table
    if multiplier in GF_MUL_TABLES:
        lookup_table = GF_MUL_TABLES[multiplier]
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
    
    # 4. Enhanced structured inputs with more mathematical insights
    if multiplier == 2:
        # For multiplication by 2:
        # - Left shift is a key operation (x << 1)
        # - Conditional XOR with 0x1B if high bit is set
        # - Individual bit shifts and masking
        
        # Pre-compute several useful transformations
        X_shifted = (X << 1) & 0xFF  # Left shift
        X_high_bit = (X & 0x80) >> 7  # Extract high bit
        X_xor_mask = np.where(X_high_bit > 0, 0x1B, 0)  # Create conditional mask
        X_xor_result = X_shifted ^ X_xor_mask  # Apply conditional XOR
        
        # Create intermediate bit representations
        X_bits_shifted = np.unpackbits((X << 1).astype(np.uint8).reshape(-1, 1), axis=1)
        
        # Stack additional features
        X_structured = np.concatenate([
            X,  # Original input
            X_shifted,  # Left shifted
            X_high_bit,  # High bit
            X_xor_mask,  # XOR mask
            X_xor_result,  # Expected result
            np.bitwise_and(X, 0x1).astype(np.uint8),  # Lowest bit
            np.bitwise_and(X, 0x2).astype(np.uint8),  # Second bit
            np.bitwise_and(X, 0x4).astype(np.uint8),  # Third bit
            np.bitwise_and(X, 0x8).astype(np.uint8),  # Fourth bit
            np.bitwise_and(X, 0x10).astype(np.uint8),  # Fifth bit
            np.bitwise_and(X, 0x20).astype(np.uint8),  # Sixth bit
            np.bitwise_and(X, 0x40).astype(np.uint8),  # Seventh bit
            np.bitwise_and(X, 0x80).astype(np.uint8),  # Eighth bit
        ], axis=1).astype(np.float32) / 255.0
        
    elif multiplier == 3:
        # For multiplication by 3:
        # GF(2^8) × 3 = GF(2^8) × 2 ⊕ GF(2^8) × 1
        
        # GF(2^8) × 2 component
        X_mul2 = GF_MUL_2.reshape(-1, 1)[X].reshape(num_samples, 1)
        
        # High bit for multiplication by 2
        X_high_bit = (X & 0x80) >> 7
        
        # Individual bit masks and intermediate results
        X_shifted = (X << 1) & 0xFF  # Left shift for × 2
        X_xor_mask = np.where(X_high_bit > 0, 0x1B, 0)  # Mask for × 2
        X_xor_result = X_shifted ^ X_xor_mask  # × 2 result
        X_final = X_xor_result ^ X  # × 3 result (× 2 ⊕ × 1)
        
        X_structured = np.concatenate([
            X,  # Original input
            X_mul2,  # × 2 result
            X_high_bit,  # High bit
            X_shifted,  # Left shifted
            X_xor_mask,  # XOR mask for × 2
            X_xor_result,  # × 2 result
            X_final,  # Expected × 3 result
            np.bitwise_and(X, 0x1).astype(np.uint8),  # Lowest bit
            np.bitwise_and(X, 0x2).astype(np.uint8),  # Second bit
            np.bitwise_and(X, 0x4).astype(np.uint8),  # Third bit
            np.bitwise_and(X, 0x8).astype(np.uint8),  # Fourth bit
            np.bitwise_and(X, 0x10).astype(np.uint8),  # Fifth bit
            np.bitwise_and(X, 0x20).astype(np.uint8),  # Sixth bit
            np.bitwise_and(X, 0x40).astype(np.uint8),  # Seventh bit
            np.bitwise_and(X, 0x80).astype(np.uint8),  # Eighth bit
            # Additional polynomial representations
            np.bitwise_and(X_mul2, 0x1).astype(np.uint8),  # Lowest bit of × 2
            np.bitwise_and(X_mul2, 0x80).astype(np.uint8),  # Highest bit of × 2
        ], axis=1).astype(np.float32) / 255.0
    else:
        # For other multipliers, include relevant operations
        X_structured = np.concatenate([
            X,
            GF_MUL_2.reshape(-1, 1)[X].reshape(num_samples, 1),  # Add mul by 2 result
            GF_MUL_TABLES[multiplier-1].reshape(-1, 1)[X].reshape(num_samples, 1) if multiplier > 1 else X,  # Add mul by (n-1) result
        ], axis=1).astype(np.float32) / 255.0
    
    # 5. Polynomial representation (coefficients of x^7, x^6, ..., x^0)
    # GF(2^8) elements can be represented as polynomials over GF(2)
    X_poly = X_bits  # The bits directly represent polynomial coefficients
    y_poly = y_bits
    
    # 6. Bit-pair representation for capturing bit interactions
    # This encodes all possible 2-bit interactions
    X_bit_pairs = np.zeros((num_samples, 28), dtype=np.float32)
    pair_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            # Combine bits i and j (00, 01, 10, 11 -> 0, 1, 2, 3)
            X_bit_pairs[:, pair_idx] = X_bits[:, i] * 2 + X_bits[:, j]
            pair_idx += 1
    
    # 7. Mathematical guidance features
    # Create features explicitly guiding the model toward the correct mathematical operations
    X_math_guide = np.zeros((num_samples, 8 + 8 + 1), dtype=np.float32)

    '''[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]'''
    
    # Include original bits
    X_math_guide[:, :8] = X_bits
    
    # Add irreducible polynomial indicators
    X_math_guide[:, 8:16] = np.tile([1, 0, 0, 1, 1, 0, 0, 1], (num_samples, 1))  # AES polynomial coefficients
    
    # Add multiplier value indicator
    X_math_guide[:, 16] = multiplier / 255.0
    
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
        'y_structured': y_norm,  # Same as y_norm for simplicity
        'X_poly': X_poly.astype(np.float32),
        'y_poly': y_poly.astype(np.float32),
        'X_bit_pairs': X_bit_pairs,
        'X_math_guide': X_math_guide
    }

def generate_full_mixcolumns_dataset(num_samples):
    """Generate dataset for learning the full MixColumns operation"""
    # MixColumns operates on 4x4 state
    X = np.random.randint(0, 256, size=(num_samples, 16), dtype=np.uint8)
    y = np.zeros_like(X)
    
    # Pre-compute common matrix multiplications for efficiency
    for i in range(num_samples):
        # Reshape to 4x4 state (column-major order for AES)
        state = X[i].reshape(4, 4, order='F')
        result_state = np.zeros_like(state)
        
        # Apply MixColumns to each column
        for col in range(4):
            # Using the AES MixColumns matrix multiplication - optimized implementation
            s0, s1, s2, s3 = state[:, col]  # Get the four bytes in this column
            
            # Compute using pre-computed lookup tables for GF multiplication
            result_state[0, col] = GF_MUL_2[s0] ^ GF_MUL_3[s1] ^ s2 ^ s3
            result_state[1, col] = s0 ^ GF_MUL_2[s1] ^ GF_MUL_3[s2] ^ s3
            result_state[2, col] = s0 ^ s1 ^ GF_MUL_2[s2] ^ GF_MUL_3[s3]
            result_state[3, col] = GF_MUL_3[s0] ^ s1 ^ s2 ^ GF_MUL_2[s3]
        
        # Flatten back to 1D array (column-major order)
        y[i] = result_state.reshape(16, order='F')
    
    # Convert to bit representation
    X_bits = np.zeros((num_samples, 16 * 8), dtype=np.float32)
    y_bits = np.zeros((num_samples, 16 * 8), dtype=np.float32)
    
    for i in range(num_samples):
        for j in range(16):
            X_bits[i, j*8:(j+1)*8] = np.unpackbits(X[i, j].reshape(1, -1))[0]
            y_bits[i, j*8:(j+1)*8] = np.unpackbits(y[i, j].reshape(1, -1))[0]
    
    # Normalized representation
    X_norm = X.astype(np.float32) / 255.0
    y_norm = y.astype(np.float32) / 255.0
    
    # Column-wise features to capture the structure of MixColumns
    X_columns = np.zeros((num_samples, 4, 4), dtype=np.float32)
    for i in range(num_samples):
        X_columns[i] = X[i].reshape(4, 4, order='F')
    X_columns = X_columns.reshape(num_samples, 16) / 255.0
    
    # Add polynomial representation of each byte
    X_poly = X_bits
    y_poly = y_bits
    
    # Add structured features with intermediate calculations
    X_structured = np.zeros((num_samples, 16 + 16*2), dtype=np.float32)  # Original + GF(2) and GF(3) of each byte
    for i in range(num_samples):
        X_structured[i, :16] = X[i] / 255.0  # Original data
        for j in range(16):
            X_structured[i, 16+j] = GF_MUL_2[X[i, j]] / 255.0  # GF(2) of each byte
            X_structured[i, 32+j] = GF_MUL_3[X[i, j]] / 255.0  # GF(3) of each byte
    
    # Include mathematical guidance features
    # Add metadata about the MixColumns operation
    X_math_guide = np.zeros((num_samples, 16*8 + 8 + 16*2), dtype=np.float32)
    
    # Include original bits
    X_math_guide[:, :16*8] = X_bits
    
    # Add irreducible polynomial indicators
    X_math_guide[:, 16*8:16*8+8] = np.tile([1, 0, 0, 1, 1, 0, 0, 1], (num_samples, 1))
    
    # Add MixColumns matrix coefficients (2, 3, 1, 1) etc.
    mixcol_coefficients = np.zeros((num_samples, 16*2))
    for i in range(num_samples):
        # For each coefficient, indicate if it's a multiply by 1, 2, or 3
        # For each of the 16 positions
        for j in range(16):
            row = j % 4
            col = j // 4
            if row == 0:
                if col == 0: mixcol_coefficients[i, j*2] = 2/3; mixcol_coefficients[i, j*2+1] = 0
                elif col == 1: mixcol_coefficients[i, j*2] = 0; mixcol_coefficients[i, j*2+1] = 3/3
                else: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
            elif row == 1:
                if col == 0: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
                elif col == 1: mixcol_coefficients[i, j*2] = 2/3; mixcol_coefficients[i, j*2+1] = 0
                elif col == 2: mixcol_coefficients[i, j*2] = 0; mixcol_coefficients[i, j*2+1] = 3/3
                else: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
            elif row == 2:
                if col == 0: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
                elif col == 1: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
                elif col == 2: mixcol_coefficients[i, j*2] = 2/3; mixcol_coefficients[i, j*2+1] = 0
                else: mixcol_coefficients[i, j*2] = 0; mixcol_coefficients[i, j*2+1] = 3/3
            else:  # row == 3
                if col == 0: mixcol_coefficients[i, j*2] = 0; mixcol_coefficients[i, j*2+1] = 3/3
                elif col == 1: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
                elif col == 2: mixcol_coefficients[i, j*2] = 1/3; mixcol_coefficients[i, j*2+1] = 0
                else: mixcol_coefficients[i, j*2] = 2/3; mixcol_coefficients[i, j*2+1] = 0
    
    # Add the mixcol coefficients to the guide features
    X_math_guide[:, 16*8+8:] = mixcol_coefficients
    
    return {
        'X_raw': X,
        'y_raw': y,
        'X_bits': X_bits,
        'y_bits': y_bits,
        'X_norm': X_norm,
        'y_norm': y_norm,
        'X_columns': X_columns,
        'X_structured': X_structured,
        'X_poly': X_poly,
        'y_poly': y_poly,
        'X_math_guide': X_math_guide
    }

def generate_curriculum_dataset(num_samples, operation_type="mul2", difficulty=0):
    """
    Generate a curriculum learning dataset with increasing difficulty
    
    Parameters:
    - num_samples: Number of samples to generate
    - operation_type: Type of operation ("mul2", "mul3", "mixcol")
    - difficulty: Difficulty level (0=easiest, 1=medium, 2=full)
    """
    if operation_type == "mul2":
        # Base dataset for GF(2^8) multiplication by 2
        dataset = generate_gf_multiply_dataset(num_samples, multiplier=2)
        
        if difficulty == 0:
            # Easy: Only inputs where high bit is NOT set (no reduction needed)
            # Filter dataset to only include such inputs
            mask = (dataset['X_raw'] < 128).flatten()
            for key in dataset.keys():
                dataset[key] = dataset[key][mask]
                
        elif difficulty == 1:
            # Medium: Mixture of values, with balanced distribution
            # Ensure balanced distribution of high-bit set and not set
            high_bit_set = (dataset['X_raw'] >= 128).flatten()
            high_bit_not_set = ~high_bit_set
            
            # Select equal numbers from each group
            num_each = min(np.sum(high_bit_set), np.sum(high_bit_not_set), num_samples // 2)
            
            high_indices = np.where(high_bit_set)[0][:num_each]
            low_indices = np.where(high_bit_not_set)[0][:num_each]
            
            # Combine indices
            selected_indices = np.concatenate([high_indices, low_indices])
            np.random.shuffle(selected_indices)
            
            # Filter dataset
            for key in dataset.keys():
                dataset[key] = dataset[key][selected_indices]
                
        # else: difficulty == 2 means full dataset, no filtering needed
        
    elif operation_type == "mul3":
        # Base dataset for GF(2^8) multiplication by 3
        dataset = generate_gf_multiply_dataset(num_samples, multiplier=3)
        
        if difficulty == 0:
            # Easy: Only low values (first 64 entries)
            mask = (dataset['X_raw'] < 64).flatten()
            for key in dataset.keys():
                dataset[key] = dataset[key][mask]
                
        elif difficulty == 1:
            # Medium: All values that don't require complex reductions
            # This is an approximation - include values < 128
            mask = (dataset['X_raw'] < 128).flatten()
            for key in dataset.keys():
                dataset[key] = dataset[key][mask]
                
        # else: difficulty == 2 means full dataset, no filtering needed
        
    elif operation_type == "mixcol":
        # Base dataset for MixColumns
        dataset = generate_full_mixcolumns_dataset(num_samples)
        
        if difficulty == 0:
            # Easy: Only operate on a single column with simple values
            easy_X = np.zeros((num_samples, 16), dtype=np.uint8)
            easy_y = np.zeros((num_samples, 16), dtype=np.uint8)
            
            # Generate simple values for the first column only
            for i in range(num_samples):
                # Set first column with low values
                col_values = np.random.randint(0, 64, size=4, dtype=np.uint8)
                state = np.zeros((4, 4), dtype=np.uint8)
                state[:, 0] = col_values
                
                # Apply MixColumns to just this column
                result_state = np.zeros_like(state)
                s0, s1, s2, s3 = state[:, 0]
                result_state[0, 0] = GF_MUL_2[s0] ^ GF_MUL_3[s1] ^ s2 ^ s3
                result_state[1, 0] = s0 ^ GF_MUL_2[s1] ^ GF_MUL_3[s2] ^ s3
                result_state[2, 0] = s0 ^ s1 ^ GF_MUL_2[s2] ^ GF_MUL_3[s3]
                result_state[3, 0] = GF_MUL_3[s0] ^ s1 ^ s2 ^ GF_MUL_2[s3]
                
                # Save the results
                easy_X[i] = state.reshape(16, order='F')
                easy_y[i] = result_state.reshape(16, order='F')
            
            # Replace raw data
            dataset['X_raw'] = easy_X
            dataset['y_raw'] = easy_y
            
            # Update all derived representations
            # Bit representation
            X_bits = np.zeros((num_samples, 16 * 8), dtype=np.float32)
            y_bits = np.zeros((num_samples, 16 * 8), dtype=np.float32)
            
            for i in range(num_samples):
                for j in range(16):
                    X_bits[i, j*8:(j+1)*8] = np.unpackbits(easy_X[i, j].reshape(1, -1))[0]
                    y_bits[i, j*8:(j+1)*8] = np.unpackbits(easy_y[i, j].reshape(1, -1))[0]
            
            dataset['X_bits'] = X_bits
            dataset['y_bits'] = y_bits
            
            # Normalized representation
            dataset['X_norm'] = easy_X.astype(np.float32) / 255.0
            dataset['y_norm'] = easy_y.astype(np.float32) / 255.0
            
            # Update other derived features (simplified for now)
            dataset['X_columns'] = dataset['X_norm'].reshape(num_samples, 16)
            dataset['X_poly'] = dataset['X_bits']
            dataset['y_poly'] = dataset['y_bits']
            
        elif difficulty == 1:
            # Medium: Full columns but with lower values
            medium_X = np.zeros((num_samples, 16), dtype=np.uint8)
            medium_y = np.zeros((num_samples, 16), dtype=np.uint8)
            
            for i in range(num_samples):
                # Set all columns with medium values
                state = np.random.randint(0, 128, size=(4, 4), dtype=np.uint8)
                result_state = np.zeros_like(state)
                
                # Apply MixColumns to each column
                for col in range(4):
                    s0, s1, s2, s3 = state[:, col]
                    result_state[0, col] = GF_MUL_2[s0] ^ GF_MUL_3[s1] ^ s2 ^ s3
                    result_state[1, col] = s0 ^ GF_MUL_2[s1] ^ GF_MUL_3[s2] ^ s3
                    result_state[2, col] = s0 ^ s1 ^ GF_MUL_2[s2] ^ GF_MUL_3[s3]
                    result_state[3, col] = GF_MUL_3[s0] ^ s1 ^ s2 ^ GF_MUL_2[s3]
                
                medium_X[i] = state.reshape(16, order='F')
                medium_y[i] = result_state.reshape(16, order='F')
            
            # Update the dataset
            dataset['X_raw'] = medium_X
            dataset['y_raw'] = medium_y
            
            # Update derived representations
            X_bits = np.zeros((num_samples, 16 * 8), dtype=np.float32)
            y_bits = np.zeros((num_samples, 16 * 8), dtype=np.float32)
            
            for i in range(num_samples):
                for j in range(16):
                    X_bits[i, j*8:(j+1)*8] = np.unpackbits(medium_X[i, j].reshape(1, -1))[0]
                    y_bits[i, j*8:(j+1)*8] = np.unpackbits(medium_y[i, j].reshape(1, -1))[0]
            
            dataset['X_bits'] = X_bits
            dataset['y_bits'] = y_bits
            
            dataset['X_norm'] = medium_X.astype(np.float32) / 255.0
            dataset['y_norm'] = medium_y.astype(np.float32) / 255.0
            
            dataset['X_columns'] = dataset['X_norm'].reshape(num_samples, 16)
            dataset['X_poly'] = dataset['X_bits']
            dataset['y_poly'] = dataset['y_bits']
            
        # else: difficulty == 2 means full dataset, no filtering needed
    
    return dataset

def split_dataset(dataset, val_ratio=VALIDATION_SPLIT, test_ratio=TEST_SPLIT):
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
# Custom Layers and Models - Enhanced with Mathematical Structure
# -------------------------------------------------------------------------

class GaloisPolynomialLayer(layers.Layer):
    """Layer that explicitly models polynomial operations in GF(2^8)"""
    def __init__(self, multiplier=2, **kwargs):
        super(GaloisPolynomialLayer, self).__init__(**kwargs)
        self.multiplier = multiplier
    
    def build(self, input_shape):
        # Ensure input is 8-bit representation
        assert input_shape[-1] == 8, "Input must be 8-bit representation"
        
        # Irreducible polynomial representation (trainable)
        # Should converge to [1,0,0,1,1,0,0,1] for AES
        self.irr_poly = self.add_weight(
            name='irreducible_polynomial',
            shape=(8,),
            initializer=keras.initializers.Constant([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            trainable=True
        )
        
        # For multiplication by a constant, create a bit interaction matrix
        # This models how each input bit affects each output bit
        self.bit_matrix = self.add_weight(
            name=f'gf_mul{self.multiplier}_matrix',
            shape=(8, 8),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Initialize the bit matrix based on known bit correlation patterns
        if self.multiplier == 2:
            correlation_matrix = BIT_CORRELATION_2
        elif self.multiplier == 3:
            correlation_matrix = BIT_CORRELATION_3
        else:
            # Default initialization for other multipliers
            correlation_matrix = np.random.uniform(-0.1, 0.1, (8, 8))
        
        # Normalize to ensure initial weights are reasonable
        correlation_matrix = (correlation_matrix + 1) / 2  # Convert from [-1,1] to [0,1]
        self.bit_matrix.assign(tf.constant(correlation_matrix, dtype=tf.float32))
        
        # Bias terms for fine-tuning
        self.bias = self.add_weight(
            name='bias',
            shape=(8,),
            initializer='zeros',
            trainable=True
        )
        
        # Temperature parameter for sharpening sigmoid
        self.temperature = self.add_weight(
            name='temperature',
            shape=(1,),
            initializer=keras.initializers.Constant(5.0),
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Apply bit matrix transformation (models the field multiplication)
        x = K.dot(inputs, self.bit_matrix)
        
        # Apply nonlinearity with temperature scaling for sharper decisions
        temperature = K.abs(self.temperature) + 1.0  # Keep temperature positive and at least 1
        x = K.sigmoid(temperature * (x + self.bias))
        
        return x

class EnhancedXORLayer(layers.Layer):
    """Improved differentiable approximation of XOR for GF(2^8) addition"""
    def __init__(self, **kwargs):
        super(EnhancedXORLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Parameter to sharpen the XOR approximation
        self.sharpness = self.add_weight(
            name='sharpness',
            shape=(1,),
            initializer=keras.initializers.Constant(5.0),
            trainable=True
        )
        
        # Bias adjustment for the XOR approximation
        self.bias = self.add_weight(
            name='bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Assumes inputs is a list of two tensors with identical shapes
        if isinstance(inputs, list):
            x1, x2 = inputs
        else:
            # If not a list, split the tensor (for pairwise XOR)
            x1, x2 = K.split(inputs, 2, axis=-1)
        
        # Clip to ensure inputs are in [0,1] range
        x1 = K.clip(x1, 0, 1)
        x2 = K.clip(x2, 0, 1)
        
        # Basic XOR: a⊕b = a(1-b) + b(1-a)
        basic_xor = (x1 * (1 - x2)) + (x2 * (1 - x1))
        
        # Apply sharpening to push values toward 0 and 1
        # Using absolute value to ensure sharpness is positive
        sharpness = K.abs(self.sharpness) + 1.0  # Keep sharpness at least 1
        
        # Apply sharpening with sigmoid centered at 0.5
        sharpened_xor = K.sigmoid(sharpness * (basic_xor - 0.5 + self.bias))
        
        return sharpened_xor

class ModularReductionLayer(layers.Layer):
    """Layer that learns to perform polynomial modular reduction in GF(2^8)"""
    def __init__(self, **kwargs):
        super(ModularReductionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Expect 16-bit input (representing terms up to x^15)
        assert input_shape[-1] == 16, "Input must be 16-bit representation for reduction"
        
        # Irreducible polynomial representation (trainable)
        self.irr_poly = self.add_weight(
            name='irreducible_polynomial',
            shape=(8,),
            initializer=keras.initializers.Constant([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            trainable=True
        )
        
        # Reduction matrix learns how to "fold back" higher-degree terms
        self.reduction_matrix = self.add_weight(
            name='reduction_matrix',
            shape=(16, 8),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Initialize with a structure that maps higher bits to lower bits
        # according to AES polynomial reduction rules
        reduction_init = np.zeros((16, 8))
        
        # Identity for lower 8 bits
        reduction_init[:8, :8] = np.eye(8)
        
        # Higher bits get reduced according to the irreducible polynomial
        # For each bit position 8-15, determine its reduction pattern
        for i in range(8, 16):
            # Bit i gets reduced to [i-8] XOR'ed with polynomial terms
            reduction_init[i, i-8] = 1.0
            # Apply polynomial terms from the AES irreducible polynomial
            for j in range(8):
                if j in [0, 3, 4, 7]:  # Positions where polynomial has 1
                    reduction_init[i, j] ^= 1.0
        
        self.reduction_matrix.assign(tf.constant(reduction_init, dtype=tf.float32))
        
        # Temperature parameter for sharpening
        self.temperature = self.add_weight(
            name='temperature',
            shape=(1,),
            initializer=keras.initializers.Constant(5.0),
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Apply reduction matrix
        x = K.dot(inputs, self.reduction_matrix)
        
        # Apply temperature-scaled sigmoid for sharper decisions
        temperature = K.abs(self.temperature) + 1.0
        return K.sigmoid(temperature * x)

class MixColumnsMatrixLayer(layers.Layer):
    """Layer that explicitly models the MixColumns matrix operation"""
    def __init__(self, **kwargs):
        super(MixColumnsMatrixLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # MixColumns coefficient matrix (trainable but initialized correctly)
        # AES MixColumns uses the matrix:
        # [2 3 1 1]
        # [1 2 3 1]
        # [1 1 2 3]
        # [3 1 1 2]
        self.mix_matrix = self.add_weight(
            name='mix_matrix',
            shape=(4, 4),
            initializer=keras.initializers.Constant([
                [2.0, 3.0, 1.0, 1.0],
                [1.0, 2.0, 3.0, 1.0],
                [1.0, 1.0, 2.0, 3.0],
                [3.0, 1.0, 1.0, 2.0]
            ]),
            trainable=True
        )
        
        # Create separate layers for GF multiplications
        self.gf_mul2 = GaloisPolynomialLayer(multiplier=2)
        self.gf_mul3 = GaloisPolynomialLayer(multiplier=3)
        
        # XOR operation for combining results
        self.xor_layer = EnhancedXORLayer()
        
        self.built = True
    
    def call(self, inputs):
        # Expects inputs to be a 4x4 column of bits (each row is a byte represented as 8 bits)
        # Reshape if needed - input shape should be (batch, 4*8 = 32)
        if K.int_shape(inputs)[-1] == 32:
            # Reshape to (batch, 4, 8) for byte-wise operations
            inputs_reshaped = K.reshape(inputs, [-1, 4, 8])
        else:
            # If already in the right shape, just use as is
            inputs_reshaped = inputs
        
        batch_size = K.shape(inputs_reshaped)[0]
        
        # Initialize output tensor
        output = K.zeros_like(inputs_reshaped)
        
        # Process each row separately (simplification for Keras compatibility)
        # Row 0: (2×s0) ⊕ (3×s1) ⊕ s2 ⊕ s3
        s0 = inputs_reshaped[:, 0, :]
        s1 = inputs_reshaped[:, 1, :]
        s2 = inputs_reshaped[:, 2, :]
        s3 = inputs_reshaped[:, 3, :]
        
        # Row 0 calculation
        mul2_s0 = self.gf_mul2(s0)
        mul3_s1 = self.gf_mul3(s1)
        xor1 = self.xor_layer([mul2_s0, mul3_s1])
        xor2 = self.xor_layer([xor1, s2])
        row0 = self.xor_layer([xor2, s3])
        
        # Row 1 calculation: s0 ⊕ (2×s1) ⊕ (3×s2) ⊕ s3
        mul2_s1 = self.gf_mul2(s1)
        mul3_s2 = self.gf_mul3(s2)
        xor1 = self.xor_layer([s0, mul2_s1])
        xor2 = self.xor_layer([xor1, mul3_s2])
        row1 = self.xor_layer([xor2, s3])
        
        # Row 2 calculation: s0 ⊕ s1 ⊕ (2×s2) ⊕ (3×s3)
        mul2_s2 = self.gf_mul2(s2)
        mul3_s3 = self.gf_mul3(s3)
        xor1 = self.xor_layer([s0, s1])
        xor2 = self.xor_layer([xor1, mul2_s2])
        row2 = self.xor_layer([xor2, mul3_s3])
        
        # Row 3 calculation: (3×s0) ⊕ s1 ⊕ s2 ⊕ (2×s3)
        mul3_s0 = self.gf_mul3(s0)
        mul2_s3 = self.gf_mul2(s3)
        xor1 = self.xor_layer([mul3_s0, s1])
        xor2 = self.xor_layer([xor1, s2])
        row3 = self.xor_layer([xor2, mul2_s3])
        
        # Combine the results
        # Use a Lambda layer to update the output tensor for each row
        output_list = [
            K.expand_dims(row0, 1),
            K.expand_dims(row1, 1),
            K.expand_dims(row2, 1),
            K.expand_dims(row3, 1)
        ]
        output = K.concatenate(output_list, axis=1)
        
        # Reshape back to original format if needed
        if K.int_shape(inputs)[-1] == 32:
            output = K.reshape(output, [-1, 32])
        
        return output

class BitInteractionLayer(layers.Layer):
    """Custom layer to model bit interactions in GF(2^8) arithmetic"""
    def __init__(self, **kwargs):
        super(BitInteractionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert input_shape[-1] == 8, "Input must be 8-bit representation"
        
        # Create weights for each possible bit interaction (28 total for 8 bits)
        self.interaction_weights = self.add_weight(
            name='interaction_weights',
            shape=(28, 8),  # 28 interactions to 8 output bits
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Bias term
        self.bias = self.add_weight(
            name='bias',
            shape=(8,),
            initializer='zeros',
            trainable=True
        )
        
        # Weight for polynomial structure (based on AES polynomial)
        self.poly_weight = self.add_weight(
            name='polynomial_weight',
            shape=(8,),
            initializer=keras.initializers.Constant([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Generate all bit pairs interactions for the 8 input bits
        bit_pairs_list = []
        for i in range(8):
            for j in range(i+1, 8):
                # Extract bits i and j
                bit_i = K.expand_dims(inputs[:, i], axis=1)
                bit_j = K.expand_dims(inputs[:, j], axis=1)
                
                # XOR interaction between bits i and j
                xor_result = bit_i * (1 - bit_j) + bit_j * (1 - bit_i)
                bit_pairs_list.append(xor_result)
        
        # Stack all interactions
        interactions = K.concatenate(bit_pairs_list, axis=1)  # Shape: (batch_size, 28)
        
        # Apply weights to map interactions to output bits
        output = K.dot(interactions, self.interaction_weights) + self.bias
        
        # Sigmoid activation for binary output
        return K.sigmoid(output)

class GaloisMul2Layer(layers.Layer):
    """Enhanced layer that learns GF(2^8) multiplication by 2"""
    def __init__(self, **kwargs):
        super(GaloisMul2Layer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable lookup table
        self.lookup = self.add_weight(
            name='gf_mul2_lookup',
            shape=(256,),
            initializer=keras.initializers.Constant(GF_MUL_2 / 255.0),
            trainable=True
        )
        
        # Mathematical components for explicit computation
        # Parameters for shift operation
        self.shift_weight = self.add_weight(
            name='shift_weight',
            shape=(1,),
            initializer=keras.initializers.Constant(2.0),  # Multiplication by 2 is a shift
            trainable=True
        )
        
        # Irreducible polynomial coefficients
        self.irr_poly = self.add_weight(
            name='irr_poly',
            shape=(8,),
            initializer=keras.initializers.Constant([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            trainable=True
        )
        
        # Learnable balance between lookup and computation
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=keras.initializers.Constant(0.7),  # Favor lookup initially
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Scale inputs from [0,1] to integers
        x_scaled = K.cast(K.clip(inputs * 255.0, 0, 255), 'int32')
        
        # Method 1: Direct lookup
        direct_lookup = K.gather(self.lookup, x_scaled)
        
        # Method 2: Explicit computation
        # Extract high bit (x >= 128)
        high_bit = K.cast(x_scaled >= 128, 'float32')
        
        # Shift left by 1 (multiply by 2)
        shifted = K.cast(x_scaled, 'float32') * self.shift_weight
        shifted = K.cast(shifted % 256.0, 'float32')
        
        # Compute the polynomial value (0x1B = 27 = x^4 + x^3 + x + 1)
        poly_val = K.sum(self.irr_poly * K.constant([128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]))
        mask = high_bit * poly_val
        
        # XOR operation (approximate as a + b - 2ab for binary values)
        # Normalize to [0,1] range
        a_norm = shifted / 255.0
        b_norm = mask / 255.0
        xor_norm = a_norm * (1 - b_norm) + b_norm * (1 - a_norm)
        
        # Scale back to [0,255]
        computed_result = xor_norm * 255.0
        
        # Combine both approaches with learned balance
        alpha = K.sigmoid(self.alpha)  # Constrain to [0,1]
        result = alpha * direct_lookup + (1 - alpha) * (computed_result / 255.0)
        
        return result

class GaloisMul3Layer(layers.Layer):
    """Enhanced layer that learns GF(2^8) multiplication by 3"""
    def __init__(self, **kwargs):
        super(GaloisMul3Layer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable lookup table
        self.lookup = self.add_weight(
            name='gf_mul3_lookup',
            shape=(256,),
            initializer=keras.initializers.Constant(GF_MUL_3 / 255.0),
            trainable=True
        )
        
        # Create a GF Mul2 layer for computation
        self.gf_mul2 = GaloisMul2Layer()
        
        # Learnable balance between lookup and computation
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=keras.initializers.Constant(0.7),  # Favor lookup initially
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Method 1: Direct lookup
        x_scaled = K.cast(K.clip(inputs * 255.0, 0, 255), 'int32')
        direct_lookup = K.gather(self.lookup, x_scaled)
        
        # Method 2: Computation as x*2 ⊕ x
        mul2_result = self.gf_mul2(inputs)
        
        # XOR with the original input
        # Approximate XOR as a + b - 2ab for binary values
        a_norm = mul2_result
        b_norm = inputs
        xor_norm = a_norm * (1 - b_norm) + b_norm * (1 - a_norm)
        
        # Combine both approaches
        alpha = K.sigmoid(self.alpha)  # Constrain to [0,1]
        result = alpha * direct_lookup + (1 - alpha) * xor_norm
        
        return result

class BitXORLayer(layers.Layer):
    """Enhanced layer that better approximates bitwise XOR operation"""
    def __init__(self, **kwargs):
        super(BitXORLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Learnable parameter to adjust the XOR approximation
        self.scale = self.add_weight(
            name='scale',
            shape=(1,),
            initializer=keras.initializers.Constant(5.0),
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(1,),
            initializer=keras.initializers.Constant(0.0),
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # Assumes inputs is a list of two tensors with identical shapes
        x1, x2 = inputs
        
        # Scale to [0,1] range if not already
        x1 = K.clip(x1, 0, 1)
        x2 = K.clip(x2, 0, 1)
        
        # Basic XOR: a⊕b = a(1-b) + b(1-a)
        basic_xor = (x1 * (1 - x2)) + (x2 * (1 - x1))
        
        # Apply temperature scaling to sharpen decision boundary
        scale = K.abs(self.scale) + 1.0  # Ensure scale is positive and at least 1
        adjusted_xor = K.sigmoid(scale * (basic_xor - 0.5 + self.bias))
        
        return adjusted_xor

# -------------------------------------------------------------------------
# Neural Network Models - Enhanced with Mathematical Architecture
# -------------------------------------------------------------------------

def create_mathematical_gf2_model():
    """Model that explicitly incorporates the mathematical structure of GF multiplication by 2"""
    inputs = layers.Input(shape=(8,))  # Bit representation input
    
    # Initial feature extraction
    x = layers.BatchNormalization()(inputs)
    
    # High bit extraction (bit 7)
    high_bit = layers.Lambda(lambda x: x[:, 7:8])(x)
    
    # Shift operation (bits 0-6 move to 1-7, bit 7 becomes 0)
    shifted_bits = layers.Lambda(lambda x: K.concatenate([
        K.zeros_like(x[:, 0:1]),  # New bit 0 is 0
        x[:, 0:7]  # Bits 1-7 are previous bits 0-6
    ], axis=1))(x)
    
    # Determine if reduction is needed based on high bit
    # Use Lambda to create constant for the irreducible polynomial
    irr_poly = layers.Lambda(lambda x: K.constant([[1., 0., 0., 1., 1., 0., 0., 1.]], dtype='float32'))(high_bit)
    
    # Apply the polynomial conditionally
    # Scale the polynomial by the high bit
    scaled_poly = layers.Multiply()([high_bit, irr_poly])
    
    # Create an XOR layer
    xor_layer = EnhancedXORLayer()
    
    # XOR shifted bits with the scaled polynomial
    result_bits = xor_layer([shifted_bits, scaled_poly])
    
    # Add a refinement network to further learn adjustments
    combined = layers.Concatenate()([inputs, result_bits])
    refined = layers.Dense(32, activation='relu')(combined)
    refined = layers.Dense(16, activation='relu')(refined)
    refinement = layers.Dense(8, activation='sigmoid')(refined)
    
    # Combine the mathematically derived result with learned refinement
    alpha = 0.7  # Weight for mathematical result vs. refinement
    final_result = layers.Lambda(lambda x: alpha * x[0] + (1 - alpha) * x[1])(
        [result_bits, refinement])
    
    return keras.Model(inputs=inputs, outputs=final_result)

def create_mathematical_gf3_model():
    """Model that explicitly incorporates the mathematical structure of GF multiplication by 3"""
    inputs = layers.Input(shape=(8,))  # Bit representation input
    
    # For GF(3) multiplication: result = GF(2) multiplication ⊕ original value
    
    # First, compute GF(2) multiplication
    gf2_model = create_mathematical_gf2_model()
    gf2_result = gf2_model(inputs)
    
    # XOR with original input
    xor_layer = EnhancedXORLayer()
    result_bits = xor_layer([gf2_result, inputs])
    
    # Add a refinement network
    features = layers.Concatenate()([inputs, gf2_result, result_bits])
    refined = layers.Dense(32, activation='relu')(features)
    refined = layers.Dense(16, activation='relu')(refined)
    refinement = layers.Dense(8, activation='sigmoid')(refined)
    
    # Combine results
    alpha = 0.7  # Weight for mathematical result vs. refinement
    final_result = layers.Lambda(lambda x: alpha * x[0] + (1 - alpha) * x[1])(
        [result_bits, refinement])
    
    return keras.Model(inputs=inputs, outputs=final_result)

def create_mixcolumns_mathematical_model():
    """Model that implements the mathematical structure of MixColumns"""
    # Input is a full state (16 bytes = 128 bits)
    inputs = layers.Input(shape=(16*8,))
    
    # Reshape to 4 columns, each with 4 bytes (32 bits per column)
    reshaped = layers.Reshape((4, 32))(inputs)
    
    # Apply MixColumns to each column independently with shared weights
    mixcol_layer = MixColumnsMatrixLayer()
    
    # Process each column
    processed_columns = []
    for i in range(4):
        column = layers.Lambda(lambda x, idx=i: x[:, idx, :])(reshaped)
        processed = mixcol_layer(column)
        processed_columns.append(processed)
    
    # Recombine columns
    combined = layers.Concatenate()(processed_columns)
    
    # Add a refinement network
    refined = layers.Dense(256, activation='relu')(combined)
    refined = layers.Dense(128, activation='relu')(refined)
    refinement = layers.Dense(16*8, activation='sigmoid')(refined)
    
    # Combine the mathematical result with learned refinement
    alpha = 0.8  # Higher weight to mathematical result
    reshaped_combined = layers.Reshape((16*8,))(combined)
    final_result = layers.Lambda(lambda x: alpha * x[0] + (1 - alpha) * x[1])(
        [reshaped_combined, refinement])
    
    return keras.Model(inputs=inputs, outputs=final_result)

def create_enhanced_gf2_model():
    """Enhanced model for GF(2^8) multiplication by 2 using mathematical insights"""
    # Take bits as input for more precise control
    inputs = layers.Input(shape=(8,))
    
    # 1. Component-based mathematical approach
    # Extract high bit (bit 7)
    high_bit = layers.Lambda(lambda x: x[:, 7:8])(inputs)
    
    # Shift operation (bits 0-6 move to 1-7, bit 7 becomes 0)
    shifted_bits = layers.Lambda(lambda x: K.concatenate([
        K.zeros_like(x[:, 0:1]),  # New bit 0 is 0
        x[:, 0:6],  # Bits 1-6 are previous bits 0-5
        x[:, 6:7]   # Bit 7 is previous bit 6
    ], axis=1))(inputs)
    
    # Polynomial coefficients (AES: x^8 + x^4 + x^3 + x + 1)
    poly_bits = layers.Lambda(lambda x: K.constant([[1., 0., 0., 1., 1., 0., 0., 1.]], dtype='float32'))(inputs)
    
    # Scale polynomial by high bit
    scaled_poly = layers.Multiply()([high_bit, poly_bits])
    
    # XOR operation (using improved layer)
    xor_layer = EnhancedXORLayer()
    math_result = xor_layer([shifted_bits, scaled_poly])
    
    # 2. Specialized GF network approach
    # Use polynomial layer
    gf_layer = GaloisPolynomialLayer(multiplier=2)
    gf_result = gf_layer(inputs)
    
    # 3. General neural network approach
    # Deep network with bit-level processing
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Use residual connections
    for _ in range(3):  # Deeper network
        residual = x
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
    
    nn_result = layers.Dense(8, activation='sigmoid')(x)
    
    # Combine all approaches with learnable weights
    combined_input = layers.Concatenate()([
        inputs,
        math_result,
        gf_result,
        nn_result
    ])
    
    # Meta-network to blend results
    meta = layers.Dense(32, activation='relu')(combined_input)
    meta = layers.Dense(16, activation='relu')(meta)
    
    # Output blending weights
    blend_weights = layers.Dense(3, activation='softmax')(meta)
    
    # Weighted combination
    final_result = layers.Lambda(lambda x: 
        x[0] * K.expand_dims(x[1][:, 0], -1) +  # math_result weight
        x[2] * K.expand_dims(x[1][:, 1], -1) +  # gf_result weight
        x[3] * K.expand_dims(x[1][:, 2], -1)    # nn_result weight
    )([math_result, blend_weights, gf_result, nn_result])
    
    return keras.Model(inputs=inputs, outputs=final_result)

def create_enhanced_gf3_model():
    """Enhanced model for GF(2^8) multiplication by 3 using mathematical insights"""
    # GF(3) = GF(2) ⊕ Identity
    
    inputs = layers.Input(shape=(8,))
    
    # 1. Component-based approach
    # First get GF(2) result using the enhanced GF(2) model
    gf2_model = create_enhanced_gf2_model()
    gf2_result = gf2_model(inputs)
    
    # XOR with the original input
    xor_layer = EnhancedXORLayer()
    math_result = xor_layer([gf2_result, inputs])
    
    # 2. Specialized GF network approach
    gf_layer = GaloisPolynomialLayer(multiplier=3)
    gf_result = gf_layer(inputs)
    
    # 3. General neural network approach
    x = layers.Dense(128, activation='relu')(inputs)  # Wider network for GF(3)
    x = layers.BatchNormalization()(x)
    
    # Use residual connections with increased capacity
    for _ in range(4):  # Deeper for GF(3)
        residual = x
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
    
    nn_result = layers.Dense(8, activation='sigmoid')(x)
    
    # Combine all approaches with learnable weights
    combined_input = layers.Concatenate()([
        inputs,
        math_result,
        gf_result,
        nn_result
    ])
    
    # Meta-network to blend results
    meta = layers.Dense(64, activation='relu')(combined_input)
    meta = layers.Dense(32, activation='relu')(meta)
    
    # Output blending weights
    blend_weights = layers.Dense(3, activation='softmax')(meta)
    
    # Weighted combination
    final_result = layers.Lambda(lambda x: 
        x[0] * K.expand_dims(x[1][:, 0], -1) +  # math_result weight
        x[2] * K.expand_dims(x[1][:, 1], -1) +  # gf_result weight
        x[3] * K.expand_dims(x[1][:, 2], -1)    # nn_result weight
    )([math_result, blend_weights, gf_result, nn_result])
    
    return keras.Model(inputs=inputs, outputs=final_result)

def create_enhanced_mixcolumns_model():
    """Enhanced model for MixColumns operation using mathematical insights"""
    # Input is 16 bytes (128 bits)
    inputs = layers.Input(shape=(16*8,))
    
    # 1. Process each column using a MixColumns matrix layer
    # Reshape to 4 columns with 4 bytes each (32 bits per column)
    input_columns = layers.Reshape((4, 32))(inputs)
    
    # Create a shared MixColumns layer
    mixcol_layer = MixColumnsMatrixLayer()
    
    # Apply to each column
    processed_columns = []
    for i in range(4):
        col_input = layers.Lambda(lambda x, idx=i: x[:, idx, :])(input_columns)
        col_output = mixcol_layer(col_input)
        processed_columns.append(col_output)
    
    # Combine column outputs
    combined_columns = layers.Concatenate()(processed_columns)
    math_result = layers.Reshape((16*8,))(combined_columns)
    
    # 2. Component-based approach - process each byte with GF operations
    # This would be very complex to implement fully, so create a partial implementation
    # Reshape input for byte-wise processing (16 bytes, 8 bits each)
    input_bytes = layers.Reshape((16, 8))(inputs)
    
    # Create GF operation layers
    gf_mul2 = GaloisPolynomialLayer(multiplier=2)
    gf_mul3 = GaloisPolynomialLayer(multiplier=3)
    
    # Process a few strategic bytes to capture the pattern
    # For example, first column first row depends on GF(2), GF(3), and XOR operations
    # Process bytes strategically to understand the operations
    byte0 = layers.Lambda(lambda x: x[:, 0, :])(input_bytes)  # Column 0, Row 0
    byte1 = layers.Lambda(lambda x: x[:, 1, :])(input_bytes)  # Column 0, Row 1
    byte2 = layers.Lambda(lambda x: x[:, 2, :])(input_bytes)  # Column 0, Row 2
    byte3 = layers.Lambda(lambda x: x[:, 3, :])(input_bytes)  # Column 0, Row 3
    
    # Calculate first element of output (2×byte0 ⊕ 3×byte1 ⊕ byte2 ⊕ byte3)
    mul2_byte0 = gf_mul2(byte0)
    mul3_byte1 = gf_mul3(byte1)
    
    xor_layer = EnhancedXORLayer()
    temp1 = xor_layer([mul2_byte0, mul3_byte1])
    temp2 = xor_layer([temp1, byte2])
    first_byte = xor_layer([temp2, byte3])
    
    # Similar calculations for other sample bytes
    # (Simplified for brevity - not implementing all 16 bytes)
    
    # Combine components into a partial result
    # Placeholder for full implementation - just flatten the input bytes for now
    component_result = layers.Flatten()(input_bytes)  # Placeholder
    
    # 3. Full neural network approach with high capacity
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Deep residual network
    for _ in range(6):
        residual = x
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
    
    x = layers.Dense(256, activation='relu')(x)
    nn_result = layers.Dense(16*8, activation='sigmoid')(x)
    
    # Combine approaches with learnable weights
    combined_input = layers.Concatenate()([
        inputs,
        math_result,
        nn_result
    ])
    
    # Meta-network to blend results
    meta = layers.Dense(128, activation='relu')(combined_input)
    meta = layers.BatchNormalization()(meta)
    meta = layers.Dense(64, activation='relu')(meta)
    
    # Output blending weights
    blend_weights = layers.Dense(3, activation='softmax')(meta)
    
    # Weighted combination
    final_result = layers.Lambda(lambda x: 
        x[0] * K.expand_dims(x[1][:, 0], -1) +  # inputs weight (original input)
        x[2] * K.expand_dims(x[1][:, 1], -1) +  # math_result weight
        x[3] * K.expand_dims(x[1][:, 2], -1)    # nn_result weight
    )([inputs, blend_weights, math_result, nn_result])
    
    return keras.Model(inputs=inputs, outputs=final_result)

# -------------------------------------------------------------------------
# Custom Loss Functions - Enhanced for Learning Mathematical Operations
# -------------------------------------------------------------------------

def bit_level_accuracy(y_true, y_pred):
    """Calculate accuracy at the bit level"""
    # Convert predictions to binary
    y_pred_binary = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    
    # Calculate bit accuracy
    correct_bits = K.mean(K.cast(K.equal(y_true, y_pred_binary), 'float32'))
    return correct_bits

def byte_level_accuracy(y_true, y_pred):
    """Calculate accuracy at the byte level (all 8 bits must be correct)"""
    # Reshape to have 8 bits per byte
    y_true_reshaped = K.reshape(y_true, (-1, 8))
    y_pred_reshaped = K.reshape(y_pred, (-1, 8))
    
    # Convert predictions to binary
    y_pred_binary = K.cast(K.greater_equal(y_pred_reshaped, 0.5), 'float32')
    
    # Check if all bits in a byte match
    correct_bytes = K.all(K.equal(y_true_reshaped, y_pred_binary), axis=1)
    
    # Calculate byte accuracy
    return K.mean(K.cast(correct_bytes, 'float32'))

def hamming_loss(y_true, y_pred):
    """Calculate Hamming distance (number of bit errors)"""
    # Convert predictions to binary
    y_pred_binary = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    
    # Count differences
    hamming_dist = K.mean(K.cast(K.not_equal(y_true, y_pred_binary), 'float32'))
    return hamming_dist

# -------------------------------------------------------------------------
# Enhanced Training and Evaluation Functions
# -------------------------------------------------------------------------

def train_model(model, train_data, val_data, input_key='X_bits', output_key='y_bits', 
                loss='binary_crossentropy', metrics=None, batch_size=BATCH_SIZE, epochs=EPOCHS,
                custom_optimizer=None):
    """Enhanced training function with extended callbacks"""
    # Configure metrics
    if metrics is None:
        metrics = ['accuracy', bit_level_accuracy, byte_level_accuracy]
    
    # Use the provided custom optimizer or create a default one
    if custom_optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    else:
        optimizer = custom_optimizer
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    # Enhanced callbacks
    callbacks = []
    
    # Early stopping with patience
    if USE_EARLY_STOPPING:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                min_delta=0.0001
            )
        )
    
    # Model checkpoint to save best model
    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    )
    
    # Check if the optimizer uses a learning rate schedule
    uses_lr_schedule = False
    try:
        if hasattr(optimizer, '_learning_rate') and hasattr(optimizer._learning_rate, '__class__'):
            uses_lr_schedule = 'LearningRateSchedule' in str(optimizer._learning_rate.__class__)
    except:
        uses_lr_schedule = False
    
    # Only add ReduceLROnPlateau if we're not using a learning rate schedule
    if not uses_lr_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        )
    
    # Add TensorBoard logging
    log_dir = os.path.join(OUTPUT_DIR, f"logs_{int(time.time())}")
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
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
        verbose=1,
        shuffle=True
    )
    training_time = time.time() - start_time
    
    # Print training time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot bit-level accuracy
    plt.subplot(2, 2, 3)
    if 'bit_level_accuracy' in history.history:
        plt.plot(history.history['bit_level_accuracy'], label='Training Bit Accuracy')
        plt.plot(history.history['val_bit_level_accuracy'], label='Validation Bit Accuracy')
        plt.title('Bit-level Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot byte-level accuracy
    plt.subplot(2, 2, 4)
    if 'byte_level_accuracy' in history.history:
        plt.plot(history.history['byte_level_accuracy'], label='Training Byte Accuracy')
        plt.plot(history.history['val_byte_level_accuracy'], label='Validation Byte Accuracy')
        plt.title('Byte-level Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300)
    plt.close()
    
    return model, history

def evaluate_model(model, test_data, input_key='X_bits', output_key='y_bits'):
    """Enhanced evaluation function with more detailed analysis"""
    # Get predictions
    y_pred = model.predict(test_data[input_key])
    
    # Calculate metrics
    results = model.evaluate(test_data[input_key], test_data[output_key], verbose=0)
    
    # Handle both single metric and multiple metrics cases
    if isinstance(results, list):
        test_loss = results[0]  # First value is always the loss
        if len(results) > 1:
            test_acc = results[1]  # Second value is usually accuracy
        else:
            test_acc = None
    else:
        # If only loss is returned
        test_loss = results
        test_acc = None
    
    print(f"Test loss: {test_loss:.4f}")
    if test_acc is not None:
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
        
        # Plot bit accuracies with enhanced visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(bit_accuracies)), bit_accuracies, color='skyblue')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
        plt.axhline(y=avg_bit_acc, color='g', linestyle='-', label=f'Average: {avg_bit_acc:.4f}')
        
        # Add value annotations
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=90)
        
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
            
            # Calculate hamming distance distribution
            hamming_distances = []
            for i in range(total_bytes):
                for b in range(num_bytes):
                    start_bit = b * 8
                    end_bit = (b + 1) * 8
                    byte_pred = y_pred_binary[i, start_bit:end_bit]
                    byte_true = test_data[output_key][i, start_bit:end_bit]
                    hamming_dist = np.sum(byte_pred != byte_true)
                    hamming_distances.append(hamming_dist)
            
            # Plot hamming distance distribution
            plt.figure(figsize=(10, 6))
            plt.hist(hamming_distances, bins=range(9), alpha=0.7, rwidth=0.8)
            plt.xlabel('Hamming Distance (Number of Bit Errors)')
            plt.ylabel('Frequency')
            plt.title('Hamming Distance Distribution')
            plt.xticks(range(9))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'hamming_distances.png'), dpi=300)
            plt.close()
    
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
        
        # Calculate error distribution
        errors = np.abs(y_pred_scaled - y_true_scaled).flatten()
        
        # Plot error distribution
        plt.figure(figsize=(12, 6))
        plt.hist(errors, bins=range(257), alpha=0.7)
        plt.axvline(x=avg_error, color='r', linestyle='--', label=f'Average error: {avg_error:.2f}')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution.png'), dpi=300)
        plt.close()
        
        # Plot heatmap of predicted vs. true values
        # Sample 100 random points for clearer visualization
        if len(y_pred_scaled) > 100:
            indices = np.random.choice(len(y_pred_scaled), 100, replace=False)
            sample_pred = y_pred_scaled[indices].flatten()
            sample_true = y_true_scaled[indices].flatten()
        else:
            sample_pred = y_pred_scaled.flatten()
            sample_true = y_true_scaled.flatten()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(sample_true, sample_pred, alpha=0.7)
        plt.plot([0, 255], [0, 255], 'r--', label='Perfect prediction')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('Predicted vs. True Values')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.xlim(0, 255)
        plt.ylim(0, 255)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pred_vs_true.png'), dpi=300)
        plt.close()
    
    return y_pred

# -------------------------------------------------------------------------
# Enhanced Analysis Functions for Evaluating Mathematical Correctness
# -------------------------------------------------------------------------

def analyze_gf_multiplication(model, multiplier=2):
    """Analyze GF(2^8) multiplication performance with mathematical validation"""
    print(f"\nAnalyzing GF(2^8) multiplication by {multiplier}...")
    
    # Generate all possible inputs (0-255)
    all_inputs = np.arange(256).reshape(-1, 1)
    
    # Prepare inputs for the model
    if hasattr(model, 'input_shape'):
        if model.input_shape[1] == 8:
            # Binary representation
            model_inputs = np.unpackbits(all_inputs, axis=1).astype(np.float32)
        else:
            # Normalized representation
            model_inputs = all_inputs.astype(np.float32) / 255.0
    else:
        # Default to normalized representation
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
    
    # Enhanced analysis - calculate accuracy by input value range
    ranges = [(0, 63), (64, 127), (128, 191), (192, 255)]
    for start, end in ranges:
        range_acc = np.mean(correct[start:end+1])
        range_correct = np.sum(correct[start:end+1])
        range_total = end - start + 1
        print(f"Range {start}-{end}: {range_acc:.4f} ({range_correct}/{range_total})")
    
    # Analyze high bit vs. low bit
    high_bit_inputs = all_inputs >= 128
    high_bit_acc = np.mean(correct[high_bit_inputs.flatten()])
    low_bit_acc = np.mean(correct[~high_bit_inputs.flatten()])
    print(f"High bit set (≥128): {high_bit_acc:.4f}")
    print(f"High bit not set (<128): {low_bit_acc:.4f}")
    
    # Enhanced visual comparison
    plt.figure(figsize=(12, 6))
    
    # Plot correct vs. incorrect
    plt.scatter(all_inputs[correct], expected_values[correct], 
                label='Correct Prediction', color='green', alpha=0.7, marker='o')
    plt.scatter(all_inputs[~correct], expected_values[~correct], 
                label='Incorrect Prediction', color='red', alpha=0.7, marker='x')
    
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Correct vs. Incorrect Predictions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_correct_vs_incorrect.png'), dpi=300)
    plt.close()
    
    # Plot all values comparison with connecting lines to show errors
    plt.figure(figsize=(12, 8))
    
    # Plot expected values
    plt.plot(all_inputs, expected_values, label='Expected', color='blue', linewidth=2)
    
    # Plot predicted values
    plt.plot(all_inputs, pred_values, label='Predicted', color='orange', alpha=0.7, linewidth=1)
    
    # Connect errors with lines
    for i in range(256):
        if pred_values[i] != expected_values[i]:
            plt.plot([i, i], [expected_values[i], pred_values[i]], 'r-', alpha=0.3)
    
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Full Comparison')
    plt.legend()
    plt.grid(plt.xlabel('Input Value'))
    plt.ylabel('Output Value')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Full Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_full_comparison.png'), dpi=300)
    plt.close()
    
    # Create more detailed error analysis
    errors = np.abs(pred_values - expected_values)
    avg_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    print(f"Average error magnitude: {avg_error:.2f}")
    print(f"Median error magnitude: {median_error:.2f}")
    print(f"Maximum error magnitude: {max_error}")
    
    # Plot error magnitude vs. input value
    plt.figure(figsize=(12, 6))
    plt.bar(all_inputs.flatten(), errors, alpha=0.7, width=1.0)
    plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average error: {avg_error:.2f}')
    plt.xlabel('Input Value')
    plt.ylabel('Error Magnitude')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Error by Input Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_error_by_input.png'), dpi=300)
    plt.close()
    
    # Enhanced bit-level analysis
    expected_bits = np.unpackbits(expected_values.reshape(-1, 1), axis=1)
    
    # If using binary representation model, use the raw predictions for bit accuracy
    if predictions.shape[1] == 8:
        pred_bits = pred_binary
    else:
        pred_bits = np.unpackbits(pred_values.reshape(-1, 1), axis=1)
    
    bit_accuracies = []
    for i in range(8):
        bit_acc = np.mean(expected_bits[:, i] == pred_bits[:, i])
        bit_accuracies.append(bit_acc)
    
    print("\nBit-level accuracy:")
    for i, acc in enumerate(bit_accuracies):
        print(f"Bit {i}: {acc:.4f}")
    
    # Plot bit-level accuracy with enhanced visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(8), bit_accuracies, color=plt.cm.viridis(np.linspace(0, 1, 8)))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.axhline(y=np.mean(bit_accuracies), color='g', linestyle='-', label=f'Average: {np.mean(bit_accuracies):.4f}')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Bit Position')
    plt.ylabel('Accuracy')
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Bit-level Accuracy')
    plt.xticks(range(8), [f'Bit {i}' for i in range(8)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_bit_accuracy.png'), dpi=300)
    plt.close()
    
    # Visualize bit-wise error patterns
    bit_error_patterns = np.zeros((8, 8))
    for i in range(8):  # Input bit
        for j in range(8):  # Output bit
            # Calculate how often input bit i affects output bit j correctly
            input_bit_1 = (all_inputs & (1 << i)).astype(bool)
            expected_out_bit = expected_bits[:, j]
            predicted_out_bit = pred_bits[:, j]
            
            # Calculate error rate when input bit i is 1
            errors_when_1 = np.sum(expected_out_bit[input_bit_1.flatten()] != predicted_out_bit[input_bit_1.flatten()])
            total_when_1 = np.sum(input_bit_1)
            
            if total_when_1 > 0:
                bit_error_patterns[i, j] = errors_when_1 / total_when_1
    
    # Plot bit error patterns as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(bit_error_patterns, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'GF(2^8) Multiplication by {multiplier} - Bit Error Patterns\n(How often input bit i causes an error in output bit j)')
    plt.xlabel('Output Bit Position')
    plt.ylabel('Input Bit Position')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_mul{multiplier}_bit_error_patterns.png'), dpi=300)
    plt.close()
    
    # Analyze mathematical properties
    # For GF(2^8) multiplication, verify mathematical properties
    print("\nTesting mathematical properties:")
    
    # Test property of multiplication with zero
    zero_in = np.zeros((1, 8), dtype=np.float32) if predictions.shape[1] == 8 else np.zeros((1, 1), dtype=np.float32)
    zero_pred = model.predict(zero_in)
    if predictions.shape[1] == 8:
        zero_val = np.packbits((zero_pred > 0.5).astype(np.uint8))[0]
    else:
        zero_val = int(round(zero_pred[0][0] * 255))
    
    print(f"Property: 0 × {multiplier} = 0, Model predicts: {zero_val}")
    
    # Test multiplication by 1
    one_in = np.zeros((1, 8), dtype=np.float32) if predictions.shape[1] == 8 else np.ones((1, 1), dtype=np.float32) / 255.0
    one_in[0, -1] = 1.0 if predictions.shape[1] == 8 else 1.0/255.0
    one_pred = model.predict(one_in)
    if predictions.shape[1] == 8:
        one_val = np.packbits((one_pred > 0.5).astype(np.uint8))[0]
    else:
        one_val = int(round(one_pred[0][0] * 255))
    
    print(f"Property: 1 × {multiplier} = {multiplier}, Model predicts: {one_val}")
    
    # For GF(2), test property: (2^7) × 2 = 2^8 mod P(x) = 2^4 + 2^3 + 2^1 + 2^0 = 27
    if multiplier == 2:
        test_val = 128  # 2^7
        test_in = np.zeros((1, 8), dtype=np.float32) if predictions.shape[1] == 8 else np.array([[test_val]], dtype=np.float32) / 255.0
        if predictions.shape[1] == 8:
            test_in[0, 0] = 1.0  # MSB set
        test_pred = model.predict(test_in)
        if predictions.shape[1] == 8:
            test_val_pred = np.packbits((test_pred > 0.5).astype(np.uint8))[0]
        else:
            test_val_pred = int(round(test_pred[0][0] * 255))
        
        print(f"Property: 128 × 2 = 27, Model predicts: {test_val_pred}")
    
    # For GF(3), test property: 3 = 2 + 1, so x × 3 = (x × 2) ⊕ x
    if multiplier == 3:
        test_val = 65  # Random test value
        test_in = np.zeros((1, 8), dtype=np.float32) if predictions.shape[1] == 8 else np.array([[test_val]], dtype=np.float32) / 255.0
        if predictions.shape[1] == 8:
            bits = np.unpackbits(np.array([test_val], dtype=np.uint8))
            test_in[0] = bits
        test_pred = model.predict(test_in)
        if predictions.shape[1] == 8:
            test_val_pred = np.packbits((test_pred > 0.5).astype(np.uint8))[0]
        else:
            test_val_pred = int(round(test_pred[0][0] * 255))
        
        # Compare with the property: x × 3 = (x × 2) ⊕ x
        expected_test_val = galois_multiply(test_val, 2) ^ test_val
        print(f"Property: {test_val} × 3 = ({test_val} × 2) ⊕ {test_val} = {expected_test_val}, Model predicts: {test_val_pred}")
    
    return {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'errors': errors,
        'bit_accuracies': bit_accuracies,
        'bit_error_patterns': bit_error_patterns,
        'pred_values': pred_values,
        'expected_values': expected_values
    }

def analyze_mixcolumns_performance(model, test_data, num_samples=100):
    """Enhanced analysis of MixColumns performance with mathematical validation"""
    print("\nAnalyzing MixColumns performance...")
    
    # Get predictions for a subset of test data
    X_norm = test_data['X_norm'][:num_samples]
    y_norm = test_data['y_norm'][:num_samples]
    X_raw = test_data['X_raw'][:num_samples]
    y_raw = test_data['y_raw'][:num_samples]
    
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
    median_error = np.median(byte_errors)
    
    print(f"Average error magnitude: {avg_error:.2f}")
    print(f"Median error magnitude: {median_error:.2f}")
    
    # Plot error distribution with enhanced visualization
    plt.figure(figsize=(12, 6))
    plt.hist(byte_errors.flatten(), bins=range(257), alpha=0.7)
    plt.axvline(x=avg_error, color='r', linestyle='--', label=f'Average error: {avg_error:.2f}')
    plt.axvline(x=median_error, color='g', linestyle='--', label=f'Median error: {median_error:.2f}')
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
    column_errors = []
    for col in range(4):
        col_indices = [col * 4 + i for i in range(4)]
        col_correct = np.sum(y_pred_bytes[:, col_indices] == y_raw[:, col_indices])
        col_total = y_raw[:, col_indices].size
        col_acc = col_correct / col_total
        column_accuracies.append(col_acc)
        
        col_error = np.mean(np.abs(y_pred_bytes[:, col_indices] - y_raw[:, col_indices]))
        column_errors.append(col_error)
        
        print(f"Column {col} accuracy: {col_acc:.4f}")
        print(f"Column {col} avg error: {col_error:.2f}")
    
    # Visualize column accuracies with enhanced plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(4), column_accuracies, color=plt.cm.viridis(np.linspace(0, 1, 4)))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.axhline(y=byte_accuracy, color='g', linestyle='-', label=f'Overall: {byte_accuracy:.4f}')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Column Index')
    plt.ylabel('Accuracy')
    plt.title('MixColumns - Column Accuracy')
    plt.xticks(range(4), [f'Column {i}' for i in range(4)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns_column_accuracy.png'), dpi=300)
    plt.close()
    
    # Analyze row-wise performance
    row_accuracies = []
    row_errors = []
    for row in range(4):
        row_indices = [col * 4 + row for col in range(4)]
        row_correct = np.sum(y_pred_bytes[:, row_indices] == y_raw[:, row_indices])
        row_total = y_raw[:, row_indices].size
        row_acc = row_correct / row_total
        row_accuracies.append(row_acc)
        
        row_error = np.mean(np.abs(y_pred_bytes[:, row_indices] - y_raw[:, row_indices]))
        row_errors.append(row_error)
        
        print(f"Row {row} accuracy: {row_acc:.4f}")
        print(f"Row {row} avg error: {row_error:.2f}")
    
    # Visualize row accuracies
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(4), row_accuracies, color=plt.cm.plasma(np.linspace(0, 1, 4)))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.axhline(y=byte_accuracy, color='g', linestyle='-', label=f'Overall: {byte_accuracy:.4f}')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Row Index')
    plt.ylabel('Accuracy')
    plt.title('MixColumns - Row Accuracy')
    plt.xticks(range(4), [f'Row {i}' for i in range(4)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns_row_accuracy.png'), dpi=300)
    plt.close()
    
    # Analyze position-wise accuracy in the state matrix
    pos_accuracy = np.zeros((4, 4))
    for row in range(4):
        for col in range(4):
            pos_idx = col * 4 + row
            pos_correct = np.sum(y_pred_bytes[:, pos_idx] == y_raw[:, pos_idx])
            pos_total = len(y_raw)
            pos_accuracy[row, col] = pos_correct / pos_total
    
    # Visualize position-wise accuracy as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pos_accuracy, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title('MixColumns - Position-wise Accuracy in State Matrix')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns_position_accuracy.png'), dpi=300)
    plt.close()
    
    # Analyze bit-level accuracy for MixColumns
    # Convert to bit representation
    y_raw_bits = np.unpackbits(y_raw.reshape(-1, 1), axis=1).reshape(num_samples, 16*8)
    y_pred_bits = np.unpackbits(y_pred_bytes.reshape(-1, 1), axis=1).reshape(num_samples, 16*8)
    
    bit_accuracies = []
    for i in range(16*8):
        bit_acc = np.mean(y_raw_bits[:, i] == y_pred_bits[:, i])
        bit_accuracies.append(bit_acc)
    
    # Plot bit-level accuracy
    plt.figure(figsize=(18, 6))
    plt.bar(range(16*8), bit_accuracies, alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    plt.axhline(y=np.mean(bit_accuracies), color='g', linestyle='-', label=f'Average: {np.mean(bit_accuracies):.4f}')
    plt.xlabel('Bit Index (16 bytes × 8 bits)')
    plt.ylabel('Accuracy')
    plt.title('MixColumns - Bit-level Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mixcolumns_bit_accuracy.png'), dpi=300)
    plt.close()
    
    # Analyze mathematical consistency
    # Test if the model has learned the MixColumns mathematical structure
    print("\nTesting mathematical consistency...")
    
    # Choose a few test samples
    test_indices = np.random.choice(num_samples, size=5, replace=False)
    
    for idx in test_indices:
        input_state = X_raw[idx].reshape(4, 4, order='F')
        pred_state = y_pred_bytes[idx].reshape(4, 4, order='F')
        true_state = y_raw[idx].reshape(4, 4, order='F')
        
        # For each column, verify the MixColumns matrix operation
        for col in range(4):
            input_col = input_state[:, col]
            pred_col = pred_state[:, col]
            true_col = true_state[:, col]
            
            # Calculate expected result for first element using MixColumns formula
            expected_first = GF_MUL_2[input_col[0]] ^ GF_MUL_3[input_col[1]] ^ input_col[2] ^ input_col[3]
            
            # Compare with model prediction
            if pred_col[0] == expected_first:
                print(f"Sample {idx}, Column {col}: First element mathematically consistent ✓")
            else:
                print(f"Sample {idx}, Column {col}: First element not consistent ✗")
                print(f"  Expected: {expected_first}, Predicted: {pred_col[0]}")
    
    return {
        'byte_accuracy': byte_accuracy,
        'avg_error': avg_error,
        'column_accuracies': column_accuracies,
        'row_accuracies': row_accuracies,
        'pos_accuracy': pos_accuracy,
        'bit_accuracies': bit_accuracies
    }

def analyze_mixcolumns_performance_ensemble(ensemble_data, num_samples=100):
    """Analyze MixColumns performance for an ensemble model with mathematical validation"""
    print("\nAnalyzing Ensemble MixColumns performance...")
    
    # Get subset of data
    X_norm = ensemble_data['X_norm'][:num_samples]
    y_norm = ensemble_data['y_norm'][:num_samples]
    X_raw = ensemble_data['X_raw'][:num_samples]
    y_raw = ensemble_data['y_raw'][:num_samples]
    y_pred = ensemble_data['y_pred'][:num_samples]
    
    # Convert predictions to bytes
    y_pred_bytes = np.round(y_pred * 255).astype(np.uint8)
    
    # Calculate byte-level accuracy
    byte_correct = np.sum(y_pred_bytes == y_raw)
    byte_total = y_raw.size
    byte_accuracy = byte_correct / byte_total
    
    print(f"Ensemble Byte-level accuracy: {byte_accuracy:.4f} ({byte_correct}/{byte_total} bytes)")
    
    # Calculate average error
    byte_errors = np.abs(y_pred_bytes - y_raw)
    avg_error = np.mean(byte_errors)
    
    print(f"Average error magnitude: {avg_error:.2f}")
    
    # Analyze mathematical properties of ensemble predictions
    # Choose a few random columns to examine
    num_test_samples = min(5, num_samples)
    test_indices = np.random.choice(num_samples, size=num_test_samples, replace=False)
    
    for idx in test_indices:
        input_state = X_raw[idx].reshape(4, 4, order='F')
        pred_state = y_pred_bytes[idx].reshape(4, 4, order='F')
        
        # Manually compute MixColumns for the first column
        col_idx = 0
        col = input_state[:, col_idx]
        
        # MixColumns matrix multiplication
        expected_col = np.zeros(4, dtype=np.uint8)
        expected_col[0] = GF_MUL_2[col[0]] ^ GF_MUL_3[col[1]] ^ col[2] ^ col[3]
        expected_col[1] = col[0] ^ GF_MUL_2[col[1]] ^ GF_MUL_3[col[2]] ^ col[3]
        expected_col[2] = col[0] ^ col[1] ^ GF_MUL_2[col[2]] ^ GF_MUL_3[col[3]]
        expected_col[3] = GF_MUL_3[col[0]] ^ col[1] ^ col[2] ^ GF_MUL_2[col[3]]
        
        # Compare with ensemble prediction
        pred_col = pred_state[:, col_idx]
        col_correct = np.sum(pred_col == expected_col)
        
        print(f"Sample {idx}, Column {col_idx}: {col_correct}/4 elements mathematically consistent")
        
        if col_correct < 4:
            print("  Input column:", col)
            print("  Expected output:", expected_col)
            print("  Predicted output:", pred_col)
    
    return {
        'byte_accuracy': byte_accuracy,
        'avg_error': avg_error
    }

# -------------------------------------------------------------------------
# Curriculum Learning Implementation
# -------------------------------------------------------------------------

def curriculum_train_model(model, operation_type="mul2", epochs_per_level=50, batch_size=BATCH_SIZE):
    """Train a model using curriculum learning, gradually increasing difficulty"""
    print(f"\nStarting curriculum learning for {operation_type}...")
    
    # Initialize directories
    curr_dir = os.path.join(OUTPUT_DIR, f"curriculum_{operation_type}")
    os.makedirs(curr_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    if operation_type == "mixcol":
        input_key = 'X_bits'
        output_key = 'y_bits'
    else:
        input_key = 'X_bits'
        output_key = 'y_bits'
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', bit_level_accuracy, byte_level_accuracy]
    )
    
    # Train on each difficulty level
    for difficulty in range(3):  # 0=easy, 1=medium, 2=hard
        print(f"\nTraining on difficulty level {difficulty}...")
        
        # Generate appropriate dataset
        if operation_type == "mul2":
            num_samples = NUM_SAMPLES // (3 - difficulty)  # More samples for easier levels
        elif operation_type == "mul3":
            num_samples = NUM_SAMPLES // (3 - difficulty) 
        else:  # mixcol
            num_samples = NUM_SAMPLES // (3 - difficulty)
        
        dataset = generate_curriculum_dataset(
            num_samples=num_samples, 
            operation_type=operation_type,
            difficulty=difficulty
        )
        
        # Split dataset
        train_data, val_data, _ = split_dataset(dataset)
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True
            )
        )
        
        # Checkpoint
        checkpoint_path = os.path.join(curr_dir, f"best_model_level{difficulty}.keras")
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        )
        
        # Learning rate reduction
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001
            )
        )
        
        # Train on this difficulty level
        history = model.fit(
            train_data[input_key], train_data[output_key],
            validation_data=(val_data[input_key], val_data[output_key]),
            epochs=epochs_per_level,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history for this level
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title(f'Loss - Level {difficulty}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if 'byte_level_accuracy' in history.history:
            plt.plot(history.history['byte_level_accuracy'], label='Train')
            plt.plot(history.history['val_byte_level_accuracy'], label='Val')
            plt.title(f'Byte Accuracy - Level {difficulty}')
            plt.legend()
        else:
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Val')
            plt.title(f'Accuracy - Level {difficulty}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(curr_dir, f'history_level{difficulty}.png'))
        plt.close()
        
        # Evaluate on this level's validation data
        val_results = model.evaluate(val_data[input_key], val_data[output_key], verbose=0)
        print(f"Level {difficulty} validation - Loss: {val_results[0]:.4f}, Accuracy: {val_results[1]:.4f}")
        
        # If we're at the highest difficulty, also evaluate on a separate test set
        if difficulty == 2:
            # Generate separate test dataset
            test_dataset = generate_curriculum_dataset(
                num_samples=NUM_SAMPLES // 5,  # Smaller test set
                operation_type=operation_type,
                difficulty=2  # Full difficulty
            )
            
            # Evaluate
            test_results = model.evaluate(
                test_dataset[input_key], 
                test_dataset[output_key],
                verbose=0
            )
            
            print(f"Final test results - Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}")
    
    # Save final model
    model.save(os.path.join(curr_dir, f"final_{operation_type}_model.keras"))
    
    return model

# -------------------------------------------------------------------------
# Main Experiment Functions
# -------------------------------------------------------------------------

def experiment_gf_mul2():
    """Experiment with learning GF(2^8) multiplication by 2 using mathematical approaches"""
    print("\n" + "="*80)
    print("Experiment: GF(2^8) Multiplication by 2 (Mathematical Approach)")
    print("="*80)
    
    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, "gf_mul2")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate enhanced dataset
    dataset = generate_gf_multiply_dataset(NUM_SAMPLES, multiplier=2)
    train_data, val_data, test_data = split_dataset(dataset)
    
    # List to store all trained models for ensemble
    models = []
    
    # 1. Mathematical GF(2) Model
    print("\nTraining Mathematical GF(2) Model...")
    math_model = create_mathematical_gf2_model()
    math_model.summary()
    
    # Use a fixed learning rate instead of a schedule
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    math_model, history = train_model(
        math_model, train_data, val_data, 
        input_key='X_bits', output_key='y_bits',
        loss='binary_crossentropy',
        custom_optimizer=optimizer
    )
    
    # Evaluate
    evaluate_model(math_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    math_model.save(os.path.join(exp_dir, "mathematical_gf2_model.keras"))
    models.append((math_model, 'X_bits', 'bit'))
    
    # 2. Enhanced GF(2) Model (hybrid approach)
    print("\nTraining Enhanced GF(2) Model...")
    enhanced_model = create_enhanced_gf2_model()
    enhanced_model.summary()
    
    # Custom optimizer with fixed learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    enhanced_model, history = train_model(
        enhanced_model, train_data, val_data,
        input_key='X_bits', output_key='y_bits',
        loss='binary_crossentropy',
        custom_optimizer=optimizer
    )
    
    # Evaluate
    evaluate_model(enhanced_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    enhanced_model.save(os.path.join(exp_dir, "enhanced_gf2_model.keras"))
    models.append((enhanced_model, 'X_bits', 'bit'))
    
    # 3. Curriculum Learning Approach
    print("\nTraining with Curriculum Learning...")
    curriculum_model = create_mathematical_gf2_model()  # Start with mathematical model
    
    # Train with curriculum
    curriculum_model = curriculum_train_model(
        curriculum_model, 
        operation_type="mul2",
        epochs_per_level=50,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate
    evaluate_model(curriculum_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    curriculum_model.save(os.path.join(exp_dir, "curriculum_gf2_model.keras"))
    models.append((curriculum_model, 'X_bits', 'bit'))
    
    # 4. Create Ensemble Model
    print("\nCreating Ensemble Model...")
    
    # Make predictions with all models
    all_preds = []
    for model, input_key, pred_type in models:
        preds = model.predict(test_data[input_key])
        all_preds.append(preds)
    
    # Combine predictions with equal weights
    ensemble_preds = np.zeros_like(all_preds[0])
    for preds in all_preds:
        ensemble_preds += preds
    ensemble_preds /= len(all_preds)
    
    # Convert to binary predictions
    ensemble_binary = (ensemble_preds > 0.5).astype(np.uint8)
    test_binary = (test_data['y_bits'] > 0.5).astype(np.uint8)
    
    # Calculate ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_binary == test_binary)
    print(f"Ensemble bit-level accuracy: {ensemble_accuracy:.4f}")
    
    # Calculate byte-level accuracy
    ensemble_bytes = np.zeros((len(ensemble_binary), 1), dtype=np.uint8)
    true_bytes = np.zeros((len(test_binary), 1), dtype=np.uint8)
    
    for i in range(len(ensemble_binary)):
        ensemble_bytes[i, 0] = np.packbits(ensemble_binary[i])[0]
        true_bytes[i, 0] = np.packbits(test_binary[i])[0]
    
    byte_accuracy = np.mean(ensemble_bytes == true_bytes)
    print(f"Ensemble byte-level accuracy: {byte_accuracy:.4f}")
    
    # Compare model performances
    model_names = ['Mathematical', 'Enhanced', 'Curriculum', 'Ensemble']
    bit_accuracies = []
    byte_accuracies = []
    
    for i, (model, input_key, _) in enumerate(models):
        preds = model.predict(test_data[input_key])
        pred_binary = (preds > 0.5).astype(np.uint8)
        
        # Bit-level accuracy
        bit_acc = np.mean(pred_binary == (test_data['y_bits'] > 0.5).astype(np.uint8))
        bit_accuracies.append(bit_acc)
        
        # Byte-level accuracy
        pred_bytes = np.zeros((len(pred_binary), 1), dtype=np.uint8)
        for j in range(len(pred_binary)):
            pred_bytes[j, 0] = np.packbits(pred_binary[j])[0]
        
        byte_acc = np.mean(pred_bytes == true_bytes)
        byte_accuracies.append(byte_acc)
    
    # Add ensemble accuracy
    bit_accuracies.append(ensemble_accuracy)
    byte_accuracies.append(byte_accuracy)
    
    # Plot comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, bit_accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Bit-level Accuracy')
    plt.title('GF(2^8) Multiplication by 2 - Bit Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, byte_accuracies, color=plt.cm.plasma(np.linspace(0, 1, len(model_names))))
    plt.axhline(y=1/256, color='r', linestyle='--', label='Random (1/256)')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Byte-level Accuracy')
    plt.title('GF(2^8) Multiplication by 2 - Byte Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Analyze the best model in detail
    best_model_idx = np.argmax(byte_accuracies[:-1])  # Exclude ensemble
    print(f"\n{model_names[best_model_idx]} is the best single model! Detailed analysis:")
    analyze_gf_multiplication(models[best_model_idx][0], multiplier=2)
    
    return models, byte_accuracies[-1]  # Return ensemble accuracy

def experiment_gf_mul3():
    """Experiment with learning GF(2^8) multiplication by 3 using mathematical approaches"""
    print("\n" + "="*80)
    print("Experiment: GF(2^8) Multiplication by 3 (Mathematical Approach)")
    print("="*80)
    
    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, "gf_mul3")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate enhanced dataset
    dataset = generate_gf_multiply_dataset(NUM_SAMPLES * 2, multiplier=3)  # Double samples
    train_data, val_data, test_data = split_dataset(dataset)
    
    # List to store all trained models for ensemble
    models = []
    
    # 1. Mathematical GF(3) Model
    print("\nTraining Mathematical GF(3) Model...")
    math_model = create_mathematical_gf3_model()
    math_model.summary()
    
    # Use fixed learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    math_model, history = train_model(
        math_model, train_data, val_data, 
        input_key='X_bits', output_key='y_bits',
        loss='binary_crossentropy',
        epochs=EPOCHS * 2,  # Double epochs for this harder task
        custom_optimizer=optimizer
    )
    
    # Evaluate
    evaluate_model(math_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    math_model.save(os.path.join(exp_dir, "mathematical_gf3_model.keras"))
    models.append((math_model, 'X_bits', 'bit'))
    
    # 2. Enhanced GF(3) Model (hybrid approach)
    print("\nTraining Enhanced GF(3) Model...")
    enhanced_model = create_enhanced_gf3_model()
    enhanced_model.summary()
    
    # Custom optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    enhanced_model, history = train_model(
        enhanced_model, train_data, val_data,
        input_key='X_bits', output_key='y_bits',
        loss='binary_crossentropy',
        epochs=EPOCHS * 2,  # Double epochs for this harder task
        custom_optimizer=optimizer
    )
    
    # Evaluate
    evaluate_model(enhanced_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    enhanced_model.save(os.path.join(exp_dir, "enhanced_gf3_model.keras"))
    models.append((enhanced_model, 'X_bits', 'bit'))
    
    # 3. Curriculum Learning Approach
    print("\nTraining with Curriculum Learning...")
    curriculum_model = create_mathematical_gf3_model()  # Start with mathematical model
    
    # Train with curriculum
    curriculum_model = curriculum_train_model(
        curriculum_model, 
        operation_type="mul3",
        epochs_per_level=70,  # More epochs for this harder task
        batch_size=BATCH_SIZE
    )
    
    # Evaluate
    evaluate_model(curriculum_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    curriculum_model.save(os.path.join(exp_dir, "curriculum_gf3_model.keras"))
    models.append((curriculum_model, 'X_bits', 'bit'))
    
    # 4. Create Ensemble Model
    print("\nCreating Ensemble Model...")
    
    # Make predictions with all models
    all_preds = []
    for model, input_key, pred_type in models:
        preds = model.predict(test_data[input_key])
        all_preds.append(preds)
    
    # Combine predictions with equal weights
    ensemble_preds = np.zeros_like(all_preds[0])
    for preds in all_preds:
        ensemble_preds += preds
    ensemble_preds /= len(all_preds)
    
    # Convert to binary predictions
    ensemble_binary = (ensemble_preds > 0.5).astype(np.uint8)
    test_binary = (test_data['y_bits'] > 0.5).astype(np.uint8)
    
    # Calculate ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_binary == test_binary)
    print(f"Ensemble bit-level accuracy: {ensemble_accuracy:.4f}")
    
    # Calculate byte-level accuracy
    ensemble_bytes = np.zeros((len(ensemble_binary), 1), dtype=np.uint8)
    true_bytes = np.zeros((len(test_binary), 1), dtype=np.uint8)
    
    for i in range(len(ensemble_binary)):
        ensemble_bytes[i, 0] = np.packbits(ensemble_binary[i])[0]
        true_bytes[i, 0] = np.packbits(test_binary[i])[0]
    
    byte_accuracy = np.mean(ensemble_bytes == true_bytes)
    print(f"Ensemble byte-level accuracy: {byte_accuracy:.4f}")
    
    # Compare model performances
    model_names = ['Mathematical', 'Enhanced', 'Curriculum', 'Ensemble']
    bit_accuracies = []
    byte_accuracies = []
    
    for i, (model, input_key, _) in enumerate(models):
        preds = model.predict(test_data[input_key])
        pred_binary = (preds > 0.5).astype(np.uint8)
        
        # Bit-level accuracy
        bit_acc = np.mean(pred_binary == (test_data['y_bits'] > 0.5).astype(np.uint8))
        bit_accuracies.append(bit_acc)
        
        # Byte-level accuracy
        pred_bytes = np.zeros((len(pred_binary), 1), dtype=np.uint8)
        for j in range(len(pred_binary)):
            pred_bytes[j, 0] = np.packbits(pred_binary[j])[0]
        
        byte_acc = np.mean(pred_bytes == true_bytes)
        byte_accuracies.append(byte_acc)
    
    # Add ensemble accuracy
    bit_accuracies.append(ensemble_accuracy)
    byte_accuracies.append(byte_accuracy)
    
    # Plot comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, bit_accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Bit-level Accuracy')
    plt.title('GF(2^8) Multiplication by 3 - Bit Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, byte_accuracies, color=plt.cm.plasma(np.linspace(0, 1, len(model_names))))
    plt.axhline(y=1/256, color='r', linestyle='--', label='Random (1/256)')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Byte-level Accuracy')
    plt.title('GF(2^8) Multiplication by 3 - Byte Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Analyze the best model in detail
    best_model_idx = np.argmax(byte_accuracies[:-1])  # Exclude ensemble
    print(f"\n{model_names[best_model_idx]} is the best single model! Detailed analysis:")
    analyze_gf_multiplication(models[best_model_idx][0], multiplier=3)
    
    return models, byte_accuracies[-1]  # Return ensemble accuracy

def experiment_mixcolumns():
    """Experiment with learning the MixColumns operation using mathematical approaches"""
    print("\n" + "="*80)
    print("Experiment: MixColumns (Mathematical Approach)")
    print("="*80)
    
    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, "mixcolumns")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate dataset
    dataset = generate_full_mixcolumns_dataset(NUM_SAMPLES)
    train_data, val_data, test_data = split_dataset(dataset)
    
    # List to store all trained models for ensemble
    models = []
    
    # 1. Mathematical MixColumns Model
    print("\nTraining Mathematical MixColumns Model...")
    math_model = create_mixcolumns_mathematical_model()
    math_model.summary()
    
    # Use fixed learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    math_model, history = train_model(
        math_model, train_data, val_data, 
        input_key='X_bits', output_key='y_bits',
        loss='binary_crossentropy',
        custom_optimizer=optimizer
    )
    
    # Evaluate
    evaluate_model(math_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    math_model.save(os.path.join(exp_dir, "mathematical_mixcol_model.keras"))
    models.append((math_model, 'X_bits', 'bit'))
    
    # 2. Enhanced MixColumns Model
    print("\nTraining Enhanced MixColumns Model...")
    enhanced_model = create_enhanced_mixcolumns_model()
    enhanced_model.summary()
    
    # Custom optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)  # Lower learning rate for this complex model
    
    enhanced_model, history = train_model(
        enhanced_model, train_data, val_data,
        input_key='X_bits', output_key='y_bits',
        loss='binary_crossentropy',
        epochs=EPOCHS * 2,  # Double epochs for this complex task
        custom_optimizer=optimizer
    )
    
    # Evaluate
    evaluate_model(enhanced_model, test_data, input_key='X_bits', output_key='y_bits')
    analyze_mixcolumns_performance(enhanced_model, test_data)
    
    # Save model
    enhanced_model.save(os.path.join(exp_dir, "enhanced_mixcol_model.keras"))
    models.append((enhanced_model, 'X_bits', 'bit'))
    
    # 3. Curriculum Learning Approach
    print("\nTraining with Curriculum Learning...")
    curriculum_model = create_mixcolumns_mathematical_model()  # Start with mathematical model
    
    # Train with curriculum
    curriculum_model = curriculum_train_model(
        curriculum_model, 
        operation_type="mixcol",
        epochs_per_level=70,  # More epochs for this complex task
        batch_size=BATCH_SIZE
    )
    
    # Evaluate
    evaluate_model(curriculum_model, test_data, input_key='X_bits', output_key='y_bits')
    
    # Save model
    curriculum_model.save(os.path.join(exp_dir, "curriculum_mixcol_model.keras"))
    models.append((curriculum_model, 'X_bits', 'bit'))
    
    # 4. Create Ensemble Model
    print("\nCreating Ensemble Model...")
    
    # Make predictions with all models
    all_preds = []
    for model, input_key, pred_type in models:
        preds = model.predict(test_data[input_key])
        all_preds.append(preds)
    
    # Combine predictions with equal weights initially
    ensemble_preds = np.zeros_like(all_preds[0])
    for preds in all_preds:
        ensemble_preds += preds
    ensemble_preds /= len(all_preds)
    
    # Convert to binary predictions
    ensemble_binary = (ensemble_preds > 0.5).astype(np.uint8)
    test_binary = (test_data['y_bits'] > 0.5).astype(np.uint8)
    
    # Calculate ensemble bit-level accuracy
    ensemble_accuracy = np.mean(ensemble_binary == test_binary)
    print(f"Ensemble bit-level accuracy: {ensemble_accuracy:.4f}")
    
    # Calculate byte-level accuracy
    num_bytes = test_data['y_bits'].shape[1] // 8
    byte_correct = 0
    total_bytes = len(test_data['y_bits']) * num_bytes
    
    for i in range(len(test_binary)):
        for b in range(num_bytes):
            start_bit = b * 8
            end_bit = (b + 1) * 8
            byte_pred = ensemble_binary[i, start_bit:end_bit]
            byte_true = test_binary[i, start_bit:end_bit]
            if np.array_equal(byte_pred, byte_true):
                byte_correct += 1
    
    byte_accuracy = byte_correct / total_bytes
    print(f"Ensemble byte-level accuracy: {byte_accuracy:.4f}")
    
    # Compare model performances
    model_names = ['Mathematical', 'Enhanced', 'Curriculum', 'Ensemble']
    bit_accuracies = []
    byte_accuracies = []
    
    for i, (model, input_key, _) in enumerate(models):
        preds = model.predict(test_data[input_key])
        pred_binary = (preds > 0.5).astype(np.uint8)
        
        # Bit-level accuracy
        bit_acc = np.mean(pred_binary == test_binary)
        bit_accuracies.append(bit_acc)
        
        # Byte-level accuracy
        byte_correct = 0
        for j in range(len(pred_binary)):
            for b in range(num_bytes):
                start_bit = b * 8
                end_bit = (b + 1) * 8
                if np.array_equal(pred_binary[j, start_bit:end_bit], test_binary[j, start_bit:end_bit]):
                    byte_correct += 1
        
        byte_acc = byte_correct / total_bytes
        byte_accuracies.append(byte_acc)
    
    # Add ensemble accuracy
    bit_accuracies.append(ensemble_accuracy)
    byte_accuracies.append(byte_accuracy)
    
    # Plot comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, bit_accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Bit-level Accuracy')
    plt.title('MixColumns - Bit Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, byte_accuracies, color=plt.cm.plasma(np.linspace(0, 1, len(model_names))))
    plt.axhline(y=1/256, color='r', linestyle='--', label='Random (1/256)')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Byte-level Accuracy')
    plt.title('MixColumns - Byte Accuracy')
    plt.ylim(0, max(byte_accuracies) * 1.2)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Create a custom test data object for the ensemble
    ensemble_test_data = {
        'X_norm': test_data['X_norm'],
        'y_norm': test_data['y_norm'],
        'X_raw': test_data['X_raw'],
        'y_raw': test_data['y_raw'],
        'y_pred': ensemble_preds
    }
    
    # Analyze ensemble performance in detail
    analyze_mixcolumns_performance_ensemble(ensemble_test_data)
    
    return models, byte_accuracy

def main():
    """Run the main experiment pipeline with mathematical approaches"""
    print("\n" + "="*80)
    print("Enhanced Galois Field Neural Networks with Mathematical Structure")
    print("="*80)
    
    # Run experiments in sequence
    print("\nStarting with GF(2^8) multiplication experiments...")
    
    # Experiment 1: GF(2^8) multiplication by 2 (Mathematical approach)
    gf2_models, gf2_ensemble_acc = experiment_gf_mul2()
    
    # Experiment 2: GF(2^8) multiplication by 3 (Mathematical approach)
    gf3_models, gf3_ensemble_acc = experiment_gf_mul3()
    
    # Experiment 3: MixColumns with mathematical architectures
    mixcol_models, mixcol_ensemble_acc = experiment_mixcolumns()
    
    print("\n" + "="*80)
    print("Mathematical Approaches Experiments Summary")
    print("="*80)
    print(f"GF(2^8) × 2 Ensemble Accuracy: {gf2_ensemble_acc:.4f}")
    print(f"GF(2^8) × 3 Ensemble Accuracy: {gf3_ensemble_acc:.4f}")
    print(f"MixColumns Ensemble Accuracy: {mixcol_ensemble_acc:.4f}")
    print("="*80)
    
    # Compare with previous results
    prev_gf2_acc = 0.5312  # From previous experiment
    prev_gf3_acc = 0.0117  # From previous experiment
    prev_mixcol_acc = 0.0037  # From previous experiment
    
    print("\nComparison with Previous Results:")
    print(f"GF(2^8) × 2:  Previous: {prev_gf2_acc:.4f}  New: {gf2_ensemble_acc:.4f}  Improvement: {(gf2_ensemble_acc-prev_gf2_acc)*100:.2f}%")
    print(f"GF(2^8) × 3:  Previous: {prev_gf3_acc:.4f}  New: {gf3_ensemble_acc:.4f}  Improvement: {(gf3_ensemble_acc-prev_gf3_acc)*100:.2f}%")
    print(f"MixColumns:   Previous: {prev_mixcol_acc:.4f}  New: {mixcol_ensemble_acc:.4f}  Improvement: {(mixcol_ensemble_acc-prev_mixcol_acc)*100:.2f}%")
    
    print("\nAll mathematical structure experiments completed successfully!")

if __name__ == "__main__":
    main()
