import numpy as np
import tensorflow as tf
from tensorflow import keras
from Crypto.Cipher import AES
from sklearn.model_selection import train_test_split
import os
import time
from tqdm import tqdm
import hashlib

def generate_fresh_aes_dataset(
    num_samples=500000, 
    bits_to_predict=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=128,
    output_dir="fresh_aes_dataset",
    seed=None
):
    """
    Generate a fresh AES encryption dataset with strict train/val/test separation.
    
    Parameters:
    - num_samples: Total number of samples to generate
    - bits_to_predict: Number of bits to predict (32 by default)
    - train_ratio, val_ratio, test_ratio: Dataset split ratios
    - batch_size: Batch size for the tf.data.Dataset
    - output_dir: Directory to save dataset files
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary containing train, validation and test datasets
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} AES encryption samples...")
    start_time = time.time()
    
    # Prepare arrays to hold the data
    X = np.zeros((num_samples, 256), dtype=np.float32)
    y = np.zeros((num_samples, bits_to_predict), dtype=np.float32)
    
    # Track unique plaintexts/keys to avoid duplicates (optional)
    unique_hashes = set()
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Generate random plaintext and key
        plaintext = np.random.bytes(16)  # 128-bit plaintext
        key = np.random.bytes(16)        # 128-bit key
        
        # Optional: Ensure uniqueness (comment out if too slow for large datasets)
        sample_hash = hashlib.md5(plaintext + b":" + key).hexdigest()
        if sample_hash in unique_hashes:
            i -= 1  # Retry this sample
            continue
        unique_hashes.add(sample_hash)
        
        # Perform AES encryption
        cipher = AES.new(key, AES.MODE_ECB)
        ciphertext = cipher.encrypt(plaintext)
        
        # Convert to binary format
        pt_bits = np.unpackbits(np.frombuffer(plaintext, dtype=np.uint8))
        key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
        ct_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
        
        # Store input (plaintext + key bits) and output (ciphertext bits)
        X[i] = np.concatenate([pt_bits, key_bits])
        y[i] = ct_bits[:bits_to_predict]
    
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds.")
    
    # Split the dataset with stratification if possible
    try:
        # For stratified split, we need a target that has reasonable class balance
        # We'll use the first bit of ciphertext as a proxy (should be ~50/50)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), 
            random_state=seed, stratify=y[:, 0]
        )
        
        # Further split temp into validation and test
        test_ratio_adjusted = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_adjusted,
            random_state=seed, stratify=y_temp[:, 0]
        )
    except ValueError:
        # If stratification fails, fall back to regular split
        print("Stratified split failed, falling back to random split")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), random_state=seed
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_adjusted, random_state=seed
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=min(100000, X_train.shape[0]), seed=seed)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Save the datasets
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    print(f"Dataset saved to {output_dir}")
    
    # Create metadata file
    with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
        f.write(f"AES Dataset Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Bits to predict: {bits_to_predict}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Validation samples: {X_val.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Input shape: {X_train.shape[1]}\n")
        f.write(f"Output shape: {y_train.shape[1]}\n")
    
    # Return the datasets
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

def verify_dataset_integrity(dataset_dir, sample_size=1000):
    """
    Verify that a generated dataset has no data leakage between splits.
    
    Parameters:
    - dataset_dir: Directory containing the dataset files
    - sample_size: Number of samples to check (for large datasets)
    
    Returns:
    - True if no leakage is detected, False otherwise
    """
    print(f"Verifying dataset integrity in {dataset_dir}...")
    
    # Load the datasets
    X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
    X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    
    # Limit sample size for very large datasets
    sample_train = min(sample_size, X_train.shape[0])
    sample_val = min(sample_size, X_val.shape[0])
    sample_test = min(sample_size, X_test.shape[0])
    
    # Create hashes of each input for comparison
    train_hashes = set()
    for i in range(sample_train):
        input_hash = hashlib.md5(X_train[i].tobytes()).hexdigest()
        train_hashes.add(input_hash)
    
    # Check for overlap with validation set
    val_overlap = 0
    for i in range(sample_val):
        input_hash = hashlib.md5(X_val[i].tobytes()).hexdigest()
        if input_hash in train_hashes:
            val_overlap += 1
    
    # Check for overlap with test set
    test_overlap = 0
    for i in range(sample_test):
        input_hash = hashlib.md5(X_test[i].tobytes()).hexdigest()
        if input_hash in train_hashes:
            test_overlap += 1
    
    # Check for overlap between validation and test
    val_hashes = set()
    for i in range(sample_val):
        input_hash = hashlib.md5(X_val[i].tobytes()).hexdigest()
        val_hashes.add(input_hash)
    
    val_test_overlap = 0
    for i in range(sample_test):
        input_hash = hashlib.md5(X_test[i].tobytes()).hexdigest()
        if input_hash in val_hashes:
            val_test_overlap += 1
    
    # Report results
    print(f"Checked {sample_train} training, {sample_val} validation, and {sample_test} test samples.")
    print(f"Training-Validation overlap: {val_overlap} samples ({val_overlap/sample_val*100:.2f}%)")
    print(f"Training-Test overlap: {test_overlap} samples ({test_overlap/sample_test*100:.2f}%)")
    print(f"Validation-Test overlap: {val_test_overlap} samples ({val_test_overlap/sample_test*100:.2f}%)")
    
    if val_overlap == 0 and test_overlap == 0 and val_test_overlap == 0:
        print("✅ No data leakage detected between dataset splits")
        return True
    else:
        print("❌ Data leakage detected between dataset splits")
        return False

def load_dataset(dataset_dir, batch_size=128):
    """
    Load a previously generated dataset.
    
    Parameters:
    - dataset_dir: Directory containing the dataset files
    - batch_size: Batch size for the tf.data.Dataset
    
    Returns:
    - Dictionary containing train, validation and test datasets
    """
    print(f"Loading dataset from {dataset_dir}...")
    
    # Load the datasets
    X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
    y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=min(100000, X_train.shape[0]))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

def analyze_dataset_distribution(dataset_dir):
    """
    Analyze the distribution of bits in the dataset to check for biases.
    
    Parameters:
    - dataset_dir: Directory containing the dataset files
    """
    print(f"Analyzing dataset distribution in {dataset_dir}...")
    
    # Load the datasets
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
    
    # Analyze bit distribution
    train_bit_means = np.mean(y_train, axis=0)
    val_bit_means = np.mean(y_val, axis=0)
    test_bit_means = np.mean(y_test, axis=0)
    
    print("Bit distribution (should be close to 0.5 for each bit):")
    print(f"Training set: Mean={np.mean(train_bit_means):.4f}, Min={np.min(train_bit_means):.4f}, Max={np.max(train_bit_means):.4f}")
    print(f"Validation set: Mean={np.mean(val_bit_means):.4f}, Min={np.min(val_bit_means):.4f}, Max={np.max(val_bit_means):.4f}")
    print(f"Test set: Mean={np.mean(test_bit_means):.4f}, Min={np.min(test_bit_means):.4f}, Max={np.max(test_bit_means):.4f}")
    
    # Check for bit correlations in training set
    print("\nChecking for bit correlations in training set...")
    corr_threshold = 0.1  # Threshold for reporting correlations
    
    # Choose a subset of samples for correlation analysis (for performance)
    sample_size = min(10000, y_train.shape[0])
    y_sample = y_train[:sample_size]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(y_sample.T)
    
    # Find high correlations (excluding self-correlations)
    high_corr_count = 0
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > corr_threshold:
                high_corr_count += 1
                if high_corr_count <= 5:  # Only show first 5 to avoid flooding output
                    print(f"High correlation between bits {i} and {j}: {corr_matrix[i, j]:.4f}")
    
    if high_corr_count > 5:
        print(f"... and {high_corr_count - 5} more correlations above threshold {corr_threshold}")
    
    if high_corr_count == 0:
        print(f"No correlations above threshold {corr_threshold} found")
    
    # Create visualization of bit distributions
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(train_bit_means))
    width = 0.25
    
    plt.bar(x - width, train_bit_means, width, label='Training')
    plt.bar(x, val_bit_means, width, label='Validation')
    plt.bar(x + width, test_bit_means, width, label='Test')
    
    plt.axhline(y=0.5, color='r', linestyle='-', label='Ideal (0.5)')
    plt.xlabel('Bit Position')
    plt.ylabel('Mean Value (0-1)')
    plt.title('Distribution of Bit Values Across Dataset Splits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(dataset_dir, "bit_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Bit distribution plot saved to {plot_path}")

# Example usage
if __name__ == "__main__":
    print("AES Dataset Generator")
    print("=" * 50)
    
    # Generate a small dataset by default when script is run directly
    dataset = generate_fresh_aes_dataset(
        num_samples=50000,  # Small default sample size
        bits_to_predict=32,
        seed=42,
        output_dir="fresh_aes_dataset"
    )
    
    # Verify dataset integrity
    verify_dataset_integrity("fresh_aes_dataset")
    
    # Analyze dataset distribution
    analyze_dataset_distribution("fresh_aes_dataset")
