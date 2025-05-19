import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Crypto.Cipher import AES
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
import hashlib
import pandas as pd
from scipy.stats import entropy

# --- Configuration ---
NUM_PLAINTEXTS = 10        # Number of different plaintexts
NUM_KEYS = 1000            # Number of different keys per plaintext
BLOCK_SIZE = 16            # AES block size in bytes (128 bits)
ANALYSIS_MODES = ["fixed_plaintext", "fixed_key", "avalanche"]
ANALYSIS_MODE = "fixed_plaintext"   # Which analysis to run
SEED = 42                  # Random seed for reproducibility
USE_VISUALIZATION = True   # Generate visualizations
OUTPUT_DIR = "aes_similarity_analysis"  # Directory for outputs
MAX_TSNE_SAMPLES = 500     # Maximum samples for t-SNE visualization to avoid memory issues

# --- Setup ---
np.random.seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Generation Functions ---
def generate_aes_datasets():
    """Generate datasets for AES analysis based on the selected mode"""
    print(f"\n{'='*50}")
    print(f"Generating AES datasets for {ANALYSIS_MODE} analysis")
    print(f"{'='*50}")
    
    if ANALYSIS_MODE == "fixed_plaintext":
        return generate_fixed_plaintext_dataset()
    elif ANALYSIS_MODE == "fixed_key":
        return generate_fixed_key_dataset()
    elif ANALYSIS_MODE == "avalanche":
        return generate_avalanche_dataset()
    else:
        raise ValueError(f"Unknown analysis mode: {ANALYSIS_MODE}")

def generate_fixed_plaintext_dataset():
    """
    Generate dataset with fixed plaintexts and varying keys.
    For each plaintext, generate NUM_KEYS different ciphertexts using different keys.
    """
    print(f"Generating {NUM_PLAINTEXTS} plaintexts with {NUM_KEYS} keys each")
    
    dataset = []
    
    # For each plaintext
    for pt_idx in tqdm(range(NUM_PLAINTEXTS)):
        # Generate a fixed plaintext
        plaintext = np.random.bytes(BLOCK_SIZE)
        
        ciphertexts = []
        keys = []
        
        # Generate NUM_KEYS different keys and encrypt the plaintext
        for key_idx in range(NUM_KEYS):
            key = np.random.bytes(BLOCK_SIZE)
            keys.append(key)
            
            cipher = AES.new(key, AES.MODE_ECB)
            ciphertext = cipher.encrypt(plaintext)
            ciphertexts.append(ciphertext)
        
        # Add to dataset
        dataset.append({
            'plaintext': plaintext,
            'plaintext_idx': pt_idx,
            'keys': keys,
            'ciphertexts': ciphertexts
        })
    
    print(f"Generated {len(dataset)} plaintext groups with {NUM_KEYS} ciphertexts each")
    return dataset

def generate_fixed_key_dataset():
    """
    Generate dataset with fixed keys and varying plaintexts.
    For each key, generate NUM_PLAINTEXTS different ciphertexts using different plaintexts.
    """
    print(f"Generating {NUM_KEYS} keys with {NUM_PLAINTEXTS} plaintexts each")
    
    dataset = []
    
    # For each key
    for key_idx in tqdm(range(NUM_KEYS)):
        # Generate a fixed key
        key = np.random.bytes(BLOCK_SIZE)
        cipher = AES.new(key, AES.MODE_ECB)
        
        plaintexts = []
        ciphertexts = []
        
        # Generate NUM_PLAINTEXTS different plaintexts and encrypt with the key
        for pt_idx in range(NUM_PLAINTEXTS):
            plaintext = np.random.bytes(BLOCK_SIZE)
            plaintexts.append(plaintext)
            
            ciphertext = cipher.encrypt(plaintext)
            ciphertexts.append(ciphertext)
        
        # Add to dataset
        dataset.append({
            'key': key,
            'key_idx': key_idx,
            'plaintexts': plaintexts,
            'ciphertexts': ciphertexts
        })
    
    print(f"Generated {len(dataset)} key groups with {NUM_PLAINTEXTS} ciphertexts each")
    return dataset

def generate_avalanche_dataset():
    """
    Generate dataset to study the avalanche effect.
    For each plaintext, generate variants with 1-bit differences and observe ciphertext changes.
    """
    print("Generating dataset for avalanche effect analysis")
    
    dataset = []
    
    # Generate a set of plaintexts and keys
    for idx in tqdm(range(NUM_PLAINTEXTS)):
        # Generate base plaintext and key
        base_plaintext = np.random.bytes(BLOCK_SIZE)
        base_key = np.random.bytes(BLOCK_SIZE)
        
        # Encrypt base plaintext with base key
        cipher = AES.new(base_key, AES.MODE_ECB)
        base_ciphertext = cipher.encrypt(base_plaintext)
        
        # Create plaintext variants (flip each bit)
        plaintext_variants = []
        pt_variant_ciphertexts = []
        
        pt_bytes = bytearray(base_plaintext)
        for byte_idx in range(BLOCK_SIZE):
            for bit_idx in range(8):
                # Flip one bit
                pt_bytes[byte_idx] ^= (1 << bit_idx)
                
                # Encrypt the variant
                variant_cipher = AES.new(base_key, AES.MODE_ECB)
                variant_ciphertext = variant_cipher.encrypt(bytes(pt_bytes))
                
                plaintext_variants.append(bytes(pt_bytes))
                pt_variant_ciphertexts.append(variant_ciphertext)
                
                # Flip the bit back for next iteration
                pt_bytes[byte_idx] ^= (1 << bit_idx)
        
        # Create key variants (flip each bit)
        key_variants = []
        key_variant_ciphertexts = []
        
        key_bytes = bytearray(base_key)
        for byte_idx in range(BLOCK_SIZE):
            for bit_idx in range(8):
                # Flip one bit
                key_bytes[byte_idx] ^= (1 << bit_idx)
                
                # Encrypt with the variant key
                variant_cipher = AES.new(bytes(key_bytes), AES.MODE_ECB)
                variant_ciphertext = variant_cipher.encrypt(base_plaintext)
                
                key_variants.append(bytes(key_bytes))
                key_variant_ciphertexts.append(variant_ciphertext)
                
                # Flip the bit back for next iteration
                key_bytes[byte_idx] ^= (1 << bit_idx)
        
        # Add to dataset
        dataset.append({
            'base_plaintext': base_plaintext,
            'base_key': base_key,
            'base_ciphertext': base_ciphertext,
            'plaintext_variants': plaintext_variants,
            'pt_variant_ciphertexts': pt_variant_ciphertexts,
            'key_variants': key_variants,
            'key_variant_ciphertexts': key_variant_ciphertexts
        })
    
    print(f"Generated {len(dataset)} avalanche effect analysis groups")
    return dataset

# --- Analysis Functions ---
def bytes_to_bits(data):
    """Convert bytes to bit array"""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def hamming_distance(bytes1, bytes2):
    """Calculate Hamming distance between two byte strings"""
    bits1 = bytes_to_bits(bytes1)
    bits2 = bytes_to_bits(bytes2)
    return np.sum(bits1 != bits2)

def calculate_similarity_matrix(ciphertexts):
    """Calculate similarity matrix for a set of ciphertexts using Hamming distance"""
    n = len(ciphertexts)
    matrix = np.zeros((n, n), dtype=np.int32)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = hamming_distance(ciphertexts[i], ciphertexts[j])
            matrix[i, j] = dist
            matrix[j, i] = dist
    
    return matrix

def analyze_fixed_plaintext_dataset(dataset):
    """Analyze dataset with fixed plaintexts and varying keys"""
    print(f"\n{'='*50}")
    print(f"Analyzing fixed plaintext dataset")
    print(f"{'='*50}")
    
    results = []
    
    # For each plaintext group
    for pt_idx, group in enumerate(dataset):
        plaintext = group['plaintext']
        ciphertexts = group['ciphertexts']
        
        # Check for collisions (identical ciphertexts)
        unique_ciphertexts = set()
        collisions = 0
        
        for ct in ciphertexts:
            ct_hash = hashlib.md5(ct).hexdigest()
            if ct_hash in unique_ciphertexts:
                collisions += 1
            unique_ciphertexts.add(ct_hash)
        
        # Calculate similarity matrix
        similarity_matrix = calculate_similarity_matrix(ciphertexts)
        
        # Analyze hamming distances
        distances = similarity_matrix[np.triu_indices(len(ciphertexts), k=1)]
        avg_distance = np.mean(distances)
        min_distance = np.min(distances) if len(distances) > 0 else 0
        max_distance = np.max(distances) if len(distances) > 0 else 0
        std_distance = np.std(distances)
        
        # Calculate expected value for random bit strings (should be ~64 for 128-bit strings)
        expected_distance = BLOCK_SIZE * 4  # 4 bits per byte on average for random strings
        
        # Calculate entropy of the distances
        hist, _ = np.histogram(distances, bins=range(0, BLOCK_SIZE*8+2))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Remove zeros for entropy calculation
        distance_entropy = entropy(probs)
        
        # Store results
        results.append({
            'plaintext_idx': pt_idx,
            'num_ciphertexts': len(ciphertexts),
            'num_unique': len(unique_ciphertexts),
            'collisions': collisions,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'std_distance': std_distance,
            'expected_distance': expected_distance,
            'distance_entropy': distance_entropy,
            'similarity_matrix': similarity_matrix,
            'distances': distances
        })
        
        # Print summary for this plaintext
        print(f"\nPlaintext {pt_idx+1}/{len(dataset)}:")
        print(f"Ciphertexts: {len(ciphertexts)}, Unique: {len(unique_ciphertexts)}, Collisions: {collisions}")
        print(f"Hamming distances: Avg={avg_distance:.2f}, Min={min_distance}, Max={max_distance}, Std={std_distance:.2f}")
        print(f"Expected random distance: {expected_distance}, Entropy: {distance_entropy:.4f}")
        
        # Visualization for this plaintext
        if USE_VISUALIZATION and pt_idx < 5:  # Limit to first 5 plaintexts to avoid too many plots
            visualize_similarity_matrix(similarity_matrix, 
                                       f"plaintext_{pt_idx+1}_similarity_matrix",
                                       f"Similarity Matrix for Plaintext {pt_idx+1}")
            
            visualize_distance_distribution(distances, 
                                           f"plaintext_{pt_idx+1}_distance_distribution",
                                           f"Hamming Distance Distribution for Plaintext {pt_idx+1}")
            
            if len(ciphertexts) <= 1000:  # TSNE works better with fewer points
                visualize_tsne_embedding(similarity_matrix, 
                                        f"plaintext_{pt_idx+1}_tsne_embedding",
                                        f"t-SNE Embedding for Plaintext {pt_idx+1}")
    
    # Aggregate analysis across all plaintexts
    all_distances = np.concatenate([r['distances'] for r in results])
    avg_all_distance = np.mean(all_distances)
    std_all_distance = np.std(all_distances)
    
    # Calculate aggregate histogram and entropy
    hist_all, _ = np.histogram(all_distances, bins=range(0, BLOCK_SIZE*8+2))
    probs_all = hist_all / np.sum(hist_all)
    probs_all = probs_all[probs_all > 0]
    all_entropy = entropy(probs_all)
    
    print(f"\nAggregate analysis across all plaintexts:")
    print(f"Total ciphertext pairs analyzed: {len(all_distances)}")
    print(f"Overall average Hamming distance: {avg_all_distance:.2f} ± {std_all_distance:.2f}")
    print(f"Overall entropy of distances: {all_entropy:.4f}")
    
    # Visualize aggregate distance distribution
    if USE_VISUALIZATION:
        visualize_distance_distribution(all_distances, 
                                      "aggregate_distance_distribution",
                                      "Aggregate Hamming Distance Distribution")
    
    return results

def analyze_fixed_key_dataset(dataset):
    """Analyze dataset with fixed keys and varying plaintexts"""
    print(f"\n{'='*50}")
    print(f"Analyzing fixed key dataset")
    print(f"{'='*50}")
    
    results = []
    
    # For each key group
    for key_idx, group in enumerate(dataset):
        key = group['key']
        plaintexts = group['plaintexts']
        ciphertexts = group['ciphertexts']
        
        # Check for collisions (identical ciphertexts)
        unique_ciphertexts = set()
        collisions = 0
        collision_details = []
        
        for i, ct in enumerate(ciphertexts):
            ct_hash = hashlib.md5(ct).hexdigest()
            if ct_hash in unique_ciphertexts:
                collisions += 1
                # Find which previous plaintext produced the same ciphertext
                for j in range(i):
                    if hashlib.md5(ciphertexts[j]).hexdigest() == ct_hash:
                        collision_details.append((i, j))
                        break
            unique_ciphertexts.add(ct_hash)
        
        # Calculate similarity matrix
        similarity_matrix = calculate_similarity_matrix(ciphertexts)
        
        # Analyze hamming distances between ciphertexts
        distances = similarity_matrix[np.triu_indices(len(ciphertexts), k=1)]
        avg_distance = np.mean(distances)
        min_distance = np.min(distances) if len(distances) > 0 else 0
        max_distance = np.max(distances) if len(distances) > 0 else 0
        std_distance = np.std(distances)
        
        # Calculate expected value for random bit strings
        expected_distance = BLOCK_SIZE * 4  # 4 bits per byte on average for random strings
        
        # Calculate entropy of the distances
        hist, _ = np.histogram(distances, bins=range(0, BLOCK_SIZE*8+2))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        distance_entropy = entropy(probs)
        
        # Analyze plaintext-ciphertext correlation
        pt_ct_correlations = []
        for i in range(len(plaintexts)):
            pt = bytes_to_bits(plaintexts[i])
            ct = bytes_to_bits(ciphertexts[i])
            # Calculate bit-wise correlation
            correlation = np.corrcoef(pt, ct)[0, 1]
            pt_ct_correlations.append(correlation)
        
        avg_correlation = np.mean(pt_ct_correlations)
        
        # Store results
        results.append({
            'key_idx': key_idx,
            'num_plaintexts': len(plaintexts),
            'num_unique_ciphertexts': len(unique_ciphertexts),
            'collisions': collisions,
            'collision_details': collision_details,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'std_distance': std_distance,
            'expected_distance': expected_distance,
            'distance_entropy': distance_entropy,
            'avg_pt_ct_correlation': avg_correlation,
            'similarity_matrix': similarity_matrix,
            'distances': distances
        })
        
        # Print summary for this key
        print(f"\nKey {key_idx+1}/{len(dataset)}:")
        print(f"Plaintexts: {len(plaintexts)}, Unique ciphertexts: {len(unique_ciphertexts)}, Collisions: {collisions}")
        print(f"Hamming distances: Avg={avg_distance:.2f}, Min={min_distance}, Max={max_distance}, Std={std_distance:.2f}")
        print(f"Expected random distance: {expected_distance}, Entropy: {distance_entropy:.4f}")
        print(f"Average plaintext-ciphertext correlation: {avg_correlation:.6f}")
        
        if collisions > 0:
            print(f"Collision details: {collision_details}")
        
        # Visualization for this key
        if USE_VISUALIZATION and key_idx < 5:  # Limit to first 5 keys
            visualize_similarity_matrix(similarity_matrix, 
                                       f"key_{key_idx+1}_similarity_matrix",
                                       f"Similarity Matrix for Key {key_idx+1}")
            
            visualize_distance_distribution(distances, 
                                           f"key_{key_idx+1}_distance_distribution",
                                           f"Hamming Distance Distribution for Key {key_idx+1}")
    
    # Aggregate analysis across all keys
    all_distances = np.concatenate([r['distances'] for r in results])
    avg_all_distance = np.mean(all_distances)
    std_all_distance = np.std(all_distances)
    
    all_correlations = [r['avg_pt_ct_correlation'] for r in results]
    avg_all_correlation = np.mean(all_correlations)
    std_all_correlation = np.std(all_correlations)
    
    # Calculate aggregate histogram and entropy
    hist_all, _ = np.histogram(all_distances, bins=range(0, BLOCK_SIZE*8+2))
    probs_all = hist_all / np.sum(hist_all)
    probs_all = probs_all[probs_all > 0]
    all_entropy = entropy(probs_all)
    
    print(f"\nAggregate analysis across all keys:")
    print(f"Total ciphertext pairs analyzed: {len(all_distances)}")
    print(f"Overall average Hamming distance: {avg_all_distance:.2f} ± {std_all_distance:.2f}")
    print(f"Overall entropy of distances: {all_entropy:.4f}")
    print(f"Overall plaintext-ciphertext correlation: {avg_all_correlation:.6f} ± {std_all_correlation:.6f}")
    
    # Visualize aggregate distance distribution
    if USE_VISUALIZATION:
        visualize_distance_distribution(all_distances, 
                                      "aggregate_key_distance_distribution",
                                      "Aggregate Hamming Distance Distribution")
    
    return results

def analyze_avalanche_dataset(dataset):
    """Analyze avalanche effect dataset"""
    print(f"\n{'='*50}")
    print(f"Analyzing avalanche effect dataset")
    print(f"{'='*50}")
    
    results = []
    
    # For each group
    for idx, group in enumerate(dataset):
        base_plaintext = group['base_plaintext']
        base_key = group['base_key']
        base_ciphertext = group['base_ciphertext']
        plaintext_variants = group['plaintext_variants']
        pt_variant_ciphertexts = group['pt_variant_ciphertexts']
        key_variants = group['key_variants']
        key_variant_ciphertexts = group['key_variant_ciphertexts']
        
        # Calculate hamming distances for plaintext variants
        pt_distances = []
        for i, variant_ct in enumerate(pt_variant_ciphertexts):
            dist = hamming_distance(base_ciphertext, variant_ct)
            pt_distances.append(dist)
        
        pt_avg_distance = np.mean(pt_distances)
        pt_std_distance = np.std(pt_distances)
        
        # Calculate hamming distances for key variants
        key_distances = []
        for i, variant_ct in enumerate(key_variant_ciphertexts):
            dist = hamming_distance(base_ciphertext, variant_ct)
            key_distances.append(dist)
        
        key_avg_distance = np.mean(key_distances)
        key_std_distance = np.std(key_distances)
        
        # Calculate theoretical ideal (assuming perfect avalanche)
        ideal_distance = BLOCK_SIZE * 4  # 50% of bits changed = 64 bits for 128-bit AES
        
        # Store results
        results.append({
            'index': idx,
            'pt_distances': pt_distances,
            'pt_avg_distance': pt_avg_distance,
            'pt_std_distance': pt_std_distance,
            'key_distances': key_distances,
            'key_avg_distance': key_avg_distance,
            'key_std_distance': key_std_distance,
            'ideal_distance': ideal_distance
        })
        
        # Print summary for this group
        print(f"\nGroup {idx+1}/{len(dataset)}:")
        print(f"Plaintext variant effect: Avg distance = {pt_avg_distance:.2f} ± {pt_std_distance:.2f}")
        print(f"Key variant effect: Avg distance = {key_avg_distance:.2f} ± {key_std_distance:.2f}")
        print(f"Ideal distance (perfect avalanche): {ideal_distance}")
        
        # Visualization for this group
        if USE_VISUALIZATION and idx < 5:  # Limit to first 5 groups
            visualize_avalanche_effect(pt_distances, key_distances, 
                                     f"group_{idx+1}_avalanche_effect",
                                     f"Avalanche Effect Analysis for Group {idx+1}")
    
    # Aggregate analysis
    all_pt_distances = np.concatenate([r['pt_distances'] for r in results])
    all_key_distances = np.concatenate([r['key_distances'] for r in results])
    
    pt_avg_all = np.mean(all_pt_distances)
    pt_std_all = np.std(all_pt_distances)
    
    key_avg_all = np.mean(all_key_distances)
    key_std_all = np.std(all_key_distances)
    
    print(f"\nAggregate avalanche effect analysis:")
    print(f"Plaintext bit flip effect: Avg={pt_avg_all:.2f} ± {pt_std_all:.2f} bits changed (out of 128)")
    print(f"Key bit flip effect: Avg={key_avg_all:.2f} ± {key_std_all:.2f} bits changed (out of 128)")
    
    # Calculate percentage of ideal
    pt_percentage = (pt_avg_all / ideal_distance) * 100
    key_percentage = (key_avg_all / ideal_distance) * 100
    
    print(f"Plaintext avalanche: {pt_percentage:.2f}% of ideal")
    print(f"Key avalanche: {key_percentage:.2f}% of ideal")
    
    # Visualize aggregate avalanche effect
    if USE_VISUALIZATION:
        visualize_avalanche_effect(all_pt_distances, all_key_distances, 
                                  "aggregate_avalanche_effect",
                                  "Aggregate Avalanche Effect Analysis")
    
    return results

def visualize_similarity_matrix(matrix, filename, title):
    """Visualize similarity matrix as heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.close()

def visualize_distance_distribution(distances, filename, title):
    """Visualize distribution of Hamming distances"""
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    bins = np.arange(0, BLOCK_SIZE*8+2) - 0.5
    plt.hist(distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=BLOCK_SIZE*4, color='red', linestyle='--', linewidth=2, 
                label=f'Expected random ({BLOCK_SIZE*4} bits)')
    plt.axvline(x=np.mean(distances), color='green', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(distances):.2f} bits)')
    plt.xlabel('Hamming Distance (bits)')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Histogram')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(distances, fill=True, color='skyblue')  # Using fill=True instead of deprecated shade=True
    plt.axvline(x=BLOCK_SIZE*4, color='red', linestyle='--', linewidth=2, 
                label=f'Expected random ({BLOCK_SIZE*4} bits)')
    plt.axvline(x=np.mean(distances), color='green', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(distances):.2f} bits)')
    plt.xlabel('Hamming Distance (bits)')
    plt.ylabel('Density')
    plt.title(f'{title} - Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.close()

def visualize_tsne_embedding(matrix, filename, title):
    """Visualize t-SNE embedding of the similarity matrix"""
    # Limit the number of samples for t-SNE to avoid memory issues
    if matrix.shape[0] > MAX_TSNE_SAMPLES:
        print(f"Reducing t-SNE samples from {matrix.shape[0]} to {MAX_TSNE_SAMPLES}")
        # Randomly select MAX_TSNE_SAMPLES samples
        indices = np.random.choice(matrix.shape[0], MAX_TSNE_SAMPLES, replace=False)
        reduced_matrix = matrix[indices][:, indices]
    else:
        reduced_matrix = matrix
    
    # Perform t-SNE - use random initialization with precomputed metric
    # Adjust perplexity based on sample size
    perplexity = min(30, reduced_matrix.shape[0]//5)  # Ensure perplexity is appropriate for sample size
    
    try:
        embedding = TSNE(n_components=2, metric='precomputed', random_state=SEED, 
                        init='random', perplexity=perplexity).fit_transform(reduced_matrix)
        
        # Plot embedding
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=50, alpha=0.6)
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")
        print("Skipping t-SNE visualization.")

def visualize_avalanche_effect(pt_distances, key_distances, filename, title):
    """Visualize avalanche effect analysis"""
    plt.figure(figsize=(15, 6))
    
    # Histogram of plaintext effect
    plt.subplot(1, 2, 1)
    bins = np.arange(0, BLOCK_SIZE*8+2) - 0.5
    plt.hist(pt_distances, bins=bins, color='blue', alpha=0.6, label='Plaintext bit flip effect')
    plt.axvline(x=BLOCK_SIZE*4, color='red', linestyle='--', linewidth=2, 
                label=f'Perfect avalanche ({BLOCK_SIZE*4} bits)')
    plt.axvline(x=np.mean(pt_distances), color='darkblue', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(pt_distances):.2f} bits)')
    plt.xlabel('Hamming Distance (bits)')
    plt.ylabel('Frequency')
    plt.title('Plaintext Bit Flip Effect')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Histogram of key effect
    plt.subplot(1, 2, 2)
    plt.hist(key_distances, bins=bins, color='green', alpha=0.6, label='Key bit flip effect')
    plt.axvline(x=BLOCK_SIZE*4, color='red', linestyle='--', linewidth=2, 
                label=f'Perfect avalanche ({BLOCK_SIZE*4} bits)')
    plt.axvline(x=np.mean(key_distances), color='darkgreen', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(key_distances):.2f} bits)')
    plt.xlabel('Hamming Distance (bits)')
    plt.ylabel('Frequency')
    plt.title('Key Bit Flip Effect')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.close()

# --- Neural Network Analysis Functions ---
def extract_features_from_ciphertexts(ciphertexts):
    """Extract features from ciphertexts for neural network analysis"""
    # Convert each ciphertext to a feature vector
    features = np.zeros((len(ciphertexts), BLOCK_SIZE*8), dtype=np.float32)
    for i, ct in enumerate(ciphertexts):
        features[i] = bytes_to_bits(ct)
    return features

def create_similarity_classifier(input_shape, num_classes):
    """Create a neural network classifier for ciphertext patterns"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_pattern_classifier(dataset, analysis_mode):
    """Train a neural network to classify ciphertexts based on their origin"""
    from sklearn.model_selection import train_test_split  # Add missing import
    
    print(f"\n{'='*50}")
    print(f"Training neural network classifier for {analysis_mode}")
    print(f"{'='*50}")
    
    if analysis_mode == "fixed_plaintext":
        # Prepare data for classification by plaintext origin
        all_features = []
        all_labels = []
        
        for group_idx, group in enumerate(dataset):
            features = extract_features_from_ciphertexts(group['ciphertexts'])
            labels = np.full(len(features), group_idx)
            
            all_features.append(features)
            all_labels.append(labels)
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        
        # Create and train model
        model = create_similarity_classifier(X_train.shape[1:], len(dataset))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {accuracy*100:.2f}%")
        
        # Check against random baseline
        random_baseline = 1.0 / len(dataset)
        print(f"Random baseline: {random_baseline*100:.2f}%")
        print(f"Improvement over random: {(accuracy - random_baseline)*100:.2f}%")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "classifier_training_history.png"), dpi=300)
        plt.close()
        
        return {
            'model': model,
            'accuracy': accuracy,
            'random_baseline': random_baseline,
            'history': history.history
        }
    
    elif analysis_mode == "fixed_key":
        # Similar implementation for fixed key analysis
        # (Code similar to fixed_plaintext case but with key indices)
        pass
    
    else:
        print(f"Neural network analysis not implemented for {analysis_mode} mode")
        return None

# --- Main Execution ---
def main():
    """Main execution function"""
    print(f"\n{'='*50}")
    print("AES Similarity Analysis Framework")
    print(f"{'='*50}")
    print(f"Analysis mode: {ANALYSIS_MODE}")
    print(f"Configuration: {NUM_PLAINTEXTS} plaintexts, {NUM_KEYS} keys")
    
    # Generate dataset
    dataset = generate_aes_datasets()
    
    # Analyze dataset based on mode
    if ANALYSIS_MODE == "fixed_plaintext":
        results = analyze_fixed_plaintext_dataset(dataset)
    elif ANALYSIS_MODE == "fixed_key":
        results = analyze_fixed_key_dataset(dataset)
    elif ANALYSIS_MODE == "avalanche":
        results = analyze_avalanche_dataset(dataset)
    
    # Optional: Neural network analysis
    try:
        if ANALYSIS_MODE in ["fixed_plaintext", "fixed_key"] and NUM_PLAINTEXTS >= 5:
            print("\nAttempting neural network analysis...")
            nn_results = train_pattern_classifier(dataset, ANALYSIS_MODE)
    except Exception as e:
        print(f"Neural network analysis failed: {e}")
        print("Continuing without neural network analysis.")
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")
    return results

if __name__ == "__main__":
    results = main()

