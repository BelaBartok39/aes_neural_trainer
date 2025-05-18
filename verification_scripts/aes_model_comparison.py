import numpy as np
import tensorflow as tf
from tensorflow import keras
from Crypto.Cipher import AES
import matplotlib.pyplot as plt
import seaborn as sns

def compare_model_with_aes(model_path, num_samples=10, bits_to_predict=32, visualize=True, output_dir="comparison_results"):
    """
    Compare model predictions with actual AES encryption.
    
    Parameters:
    - model_path: Path to the saved model
    - num_samples: Number of random samples to test
    - bits_to_predict: Number of bits the model was trained to predict
    - visualize: Whether to create visualizations
    - output_dir: Directory to save outputs
    
    Returns:
    - Dictionary with comparison statistics and sample data
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # Initialize results containers
    results = {
        'plaintexts': [],
        'keys': [],
        'actual_ciphertexts': [],
        'predicted_bits': [],
        'actual_bits': [],
        'bit_matches': [],
        'bit_accuracies': []
    }
    
    print(f"\nGenerating {num_samples} random plaintext/key samples...")
    
    # Process each sample
    for i in range(num_samples):
        # 1. Generate random plaintext and key
        plaintext = np.random.bytes(16)  # 128-bit plaintext
        key = np.random.bytes(16)        # 128-bit key
        
        # Store original values
        results['plaintexts'].append(plaintext)
        results['keys'].append(key)
        
        # 2. Perform actual AES encryption
        cipher = AES.new(key, AES.MODE_ECB)
        actual_ciphertext = cipher.encrypt(plaintext)
        results['actual_ciphertexts'].append(actual_ciphertext)
        
        # 3. Convert data to binary format for model input
        pt_bits = np.unpackbits(np.frombuffer(plaintext, dtype=np.uint8))
        key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
        model_input = np.concatenate([pt_bits, key_bits])
        
        # 4. Convert actual ciphertext to binary
        actual_bits_full = np.unpackbits(np.frombuffer(actual_ciphertext, dtype=np.uint8))
        actual_bits = actual_bits_full[:bits_to_predict]
        
        # 5. Make prediction with model
        prediction = model.predict(np.array([model_input]), verbose=0)[0]
        predicted_bits = (prediction > 0.5).astype(int)
        
        # 6. Compare prediction with actual
        bit_matches = (predicted_bits == actual_bits)
        bit_accuracy = np.mean(bit_matches)
        
        # Store results
        results['predicted_bits'].append(predicted_bits)
        results['actual_bits'].append(actual_bits)
        results['bit_matches'].append(bit_matches)
        results['bit_accuracies'].append(bit_accuracy)
        
        print(f"Sample {i+1}: {bit_accuracy*100:.2f}% bit accuracy ({np.sum(bit_matches)}/{len(bit_matches)} bits correct)")
    
    # Calculate overall statistics
    overall_accuracy = np.mean(results['bit_accuracies'])
    std_accuracy = np.std(results['bit_accuracies'])
    print(f"\nOverall accuracy: {overall_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    # Accuracy by bit position
    bit_position_accuracy = np.mean([matches for matches in results['bit_matches']], axis=0)
    
    # Visualizations
    if visualize:
        # Visualization 1: Actual vs. Predicted bits for each sample
        for i in range(min(5, num_samples)):  # Show up to 5 samples
            plt.figure(figsize=(12, 5))
            
            # Plot actual bits
            plt.subplot(2, 1, 1)
            plt.imshow([results['actual_bits'][i]], aspect='auto', cmap='Blues')
            plt.title(f'Sample {i+1} - Actual Bits')
            plt.ylabel('Sample')
            plt.yticks([])
            
            # Plot predicted bits
            plt.subplot(2, 1, 2)
            plt.imshow([results['predicted_bits'][i]], aspect='auto', cmap='Blues')
            plt.title(f'Sample {i+1} - Predicted Bits')
            plt.ylabel('Sample')
            plt.xlabel('Bit Position')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}_comparison.png"), dpi=300)
            plt.close()
        
        # Visualization 2: Bit position accuracy
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(bit_position_accuracy)), bit_position_accuracy)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random guessing')
        plt.xlabel('Bit Position')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Bit Position Across All Samples')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bit_position_accuracy.png"), dpi=300)
        plt.close()
        
        # Visualization 3: Distribution of sample accuracies
        plt.figure(figsize=(10, 6))
        plt.hist(results['bit_accuracies'], bins=10, alpha=0.7)
        plt.axvline(x=overall_accuracy, color='r', linestyle='--', 
                   label=f'Mean Accuracy: {overall_accuracy:.4f}')
        plt.axvline(x=0.5, color='g', linestyle='--', 
                   label='Random Guessing: 0.5')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Sample Accuracies')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_distribution.png"), dpi=300)
        plt.close()
        
        # Visualization 4: Heatmap of all samples
        plt.figure(figsize=(14, 8))
        
        # Create a combined array of all actual bits
        all_actual = np.array(results['actual_bits'])
        all_predicted = np.array(results['predicted_bits'])
        
        # Plot actual vs predicted for all samples
        plt.subplot(2, 1, 1)
        plt.imshow(all_actual, aspect='auto', cmap='Blues')
        plt.title('All Samples - Actual Bits')
        plt.ylabel('Sample')
        
        plt.subplot(2, 1, 2)
        plt.imshow(all_predicted, aspect='auto', cmap='Blues')
        plt.title('All Samples - Predicted Bits')
        plt.ylabel('Sample')
        plt.xlabel('Bit Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_samples_comparison.png"), dpi=300)
        plt.close()
    
    # Detailed statistical analysis
    print("\nDetailed Analysis:")
    print(f"  • Number of samples: {num_samples}")
    print(f"  • Bits predicted per sample: {bits_to_predict}")
    print(f"  • Overall bit accuracy: {overall_accuracy:.6f}")
    print(f"  • Standard deviation: {std_accuracy:.6f}")
    
    # Find best and worst bit positions
    best_bit = np.argmax(bit_position_accuracy)
    worst_bit = np.argmin(bit_position_accuracy)
    print(f"  • Best bit position: {best_bit} with accuracy {bit_position_accuracy[best_bit]:.6f}")
    print(f"  • Worst bit position: {worst_bit} with accuracy {bit_position_accuracy[worst_bit]:.6f}")
    
    # Count bits performing above random chance (0.51)
    better_than_random = np.sum(bit_position_accuracy > 0.51)
    print(f"  • Bits performing better than random chance (>51%): {better_than_random}/{len(bit_position_accuracy)}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'std_accuracy': std_accuracy,
        'bit_position_accuracy': bit_position_accuracy,
        'sample_accuracies': results['bit_accuracies'],
        'best_bit': best_bit,
        'worst_bit': worst_bit,
        'better_than_random': better_than_random
    }

def test_specific_plaintexts(model_path, plaintexts, keys=None, bits_to_predict=32, output_dir="specific_test_results"):
    """
    Test the model on specific plaintexts and keys.
    
    Parameters:
    - model_path: Path to the saved model
    - plaintexts: List of plaintexts to test (each should be bytes of length 16)
    - keys: List of keys to test (each should be bytes of length 16). If None, a random key will be used for each plaintext.
    - bits_to_predict: Number of bits the model was trained to predict
    - output_dir: Directory to save outputs
    
    Returns:
    - Dictionary with results for each plaintext/key pair
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # Initialize results
    results = []
    
    # Ensure keys is a list
    if keys is None:
        keys = [np.random.bytes(16) for _ in range(len(plaintexts))]
    elif isinstance(keys, bytes):
        keys = [keys] * len(plaintexts)
    
    # Process each plaintext/key pair
    for i, (plaintext, key) in enumerate(zip(plaintexts, keys)):
        # Check plaintext and key are bytes and have the right length
        if not isinstance(plaintext, bytes) or len(plaintext) != 16:
            plaintext = plaintext.ljust(16, b'\x00')[:16]  # Pad or truncate to 16 bytes
        
        if not isinstance(key, bytes) or len(key) != 16:
            key = key.ljust(16, b'\x00')[:16]  # Pad or truncate to 16 bytes
        
        # Perform actual AES encryption
        cipher = AES.new(key, AES.MODE_ECB)
        actual_ciphertext = cipher.encrypt(plaintext)
        
        # Convert to binary format for model
        pt_bits = np.unpackbits(np.frombuffer(plaintext, dtype=np.uint8))
        key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
        model_input = np.concatenate([pt_bits, key_bits])
        
        # Convert actual ciphertext to binary
        actual_bits_full = np.unpackbits(np.frombuffer(actual_ciphertext, dtype=np.uint8))
        actual_bits = actual_bits_full[:bits_to_predict]
        
        # Make prediction
        prediction = model.predict(np.array([model_input]), verbose=0)[0]
        predicted_bits = (prediction > 0.5).astype(int)
        
        # Convert predicted bits back to bytes (for display)
        # Pad with zeros to make a multiple of 8 bits if needed
        pad_length = (8 - len(predicted_bits) % 8) % 8
        padded_bits = np.pad(predicted_bits, (0, pad_length))
        
        # Reshape and convert to bytes
        bytes_array = np.packbits(padded_bits)
        predicted_bytes = bytes(bytes_array)
        
        # Compare prediction with actual
        bit_matches = (predicted_bits == actual_bits)
        bit_accuracy = np.mean(bit_matches)
        
        # Store results
        result = {
            'plaintext': plaintext,
            'plaintext_hex': plaintext.hex(),
            'key': key,
            'key_hex': key.hex(),
            'actual_ciphertext': actual_ciphertext,
            'actual_ciphertext_hex': actual_ciphertext.hex(),
            'predicted_bits': predicted_bits,
            'predicted_bytes': predicted_bytes,
            'predicted_bytes_hex': predicted_bytes.hex(),
            'actual_bits': actual_bits,
            'bit_matches': bit_matches,
            'bit_accuracy': bit_accuracy
        }
        results.append(result)
        
        # Print results
        print(f"\nTest {i+1}:")
        print(f"Plaintext (hex): {plaintext.hex()}")
        print(f"Key (hex): {key.hex()}")
        print(f"Actual ciphertext (hex): {actual_ciphertext.hex()}")
        print(f"Predicted first {bits_to_predict} bits as bytes (hex): {predicted_bytes.hex()}")
        print(f"Bit accuracy: {bit_accuracy*100:.2f}% ({np.sum(bit_matches)}/{len(bit_matches)} bits correct)")
        
        # Visualize the comparison
        plt.figure(figsize=(12, 6))
        
        # Create a comparison array: 1=match, 0=mismatch
        comparison = np.zeros((3, len(actual_bits)))
        comparison[0, :] = actual_bits
        comparison[1, :] = predicted_bits
        comparison[2, :] = bit_matches
        
        # Labels for the rows
        row_labels = ['Actual', 'Predicted', 'Match']
        
        # Plot heatmap
        sns.heatmap(comparison, cmap='Blues', cbar=False, 
                   yticklabels=row_labels, xticklabels=False)
        plt.title(f'Bit Comparison - Accuracy: {bit_accuracy*100:.2f}%')
        plt.xlabel('Bit Position')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"test_{i+1}_comparison.png"), dpi=300)
        plt.close()
    
    # Calculate overall statistics
    overall_accuracy = np.mean([result['bit_accuracy'] for result in results])
    print(f"\nOverall accuracy across all tests: {overall_accuracy*100:.2f}%")
    
    return {
        'results': results,
        'overall_accuracy': overall_accuracy
    }

def convert_model_prediction_to_bytes(predicted_bits):
    """
    Convert model's predicted bits to a byte representation.
    
    Parameters:
    - predicted_bits: Numpy array of bits (0s and 1s)
    
    Returns:
    - Bytes object representing the predicted bits
    """
    # Ensure we have a multiple of 8 bits (required for byte conversion)
    pad_length = (8 - len(predicted_bits) % 8) % 8
    if pad_length > 0:
        padded_bits = np.pad(predicted_bits, (0, pad_length))
    else:
        padded_bits = predicted_bits
    
    # Convert to bytes
    bytes_array = np.packbits(padded_bits)
    return bytes(bytes_array)

# Example usage:
if __name__ == "__main__":
    # Example calls (uncomment to use):
    """
    # 1. Compare model with actual AES on random samples
    compare_model_with_aes(
        model_path="aes_nn_models/aes_cnn_model_best.keras",
        num_samples=20,
        bits_to_predict=32
    )
    
    # 2. Test specific plaintext/key pairs
    test_specific_plaintexts(
        model_path="aes_nn_models/aes_cnn_model_best.keras",
        plaintexts=[
            b'This is a test!!',
            b'Another example.',
            b'Hello, world!!!',
        ],
        keys=b'SuperSecretKey!!'  # Same key for all plaintexts
    )
    """
