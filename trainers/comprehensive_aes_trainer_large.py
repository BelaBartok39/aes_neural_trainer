import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
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
from sklearn.model_selection import train_test_split

# --- Enhanced Configuration ---
NUM_PLAINTEXTS = 100       # Increased from 10 to 100
NUM_KEYS = 5000            # Increased from 1000 to 5000
BLOCK_SIZE = 16            # AES block size in bytes (128 bits)
ANALYSIS_MODES = ["fixed_plaintext", "fixed_key", "avalanche"]
ANALYSIS_MODE = "fixed_plaintext"
SEED = 42
USE_VISUALIZATION = True
OUTPUT_DIR = "aes_large_scale_analysis"
MAX_TSNE_SAMPLES = 1000
NEURAL_NETWORK_SIZE = "large"  # Options: "small", "medium", "large", "xlarge"
TRAINING_EPOCHS = 100           # Increased from 20 to 100
BATCH_SIZE = 64
USE_EARLY_STOPPING = True
PATIENCE = 20
USE_DATA_AUGMENTATION = True    # Enable data augmentation
USE_BIT_IMPORTANCE_ANALYSIS = True  # Analyze which bits are most important
USE_CROSS_VALIDATION = True    # Use cross-validation for more robust results

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Data Generation Functions ---
def generate_aes_datasets():
    """Generate datasets for AES analysis based on the selected mode"""
    print(f"\n{'='*50}")
    print(f"Generating large-scale AES datasets for {ANALYSIS_MODE} analysis")
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
    print(f"Total ciphertexts: {NUM_PLAINTEXTS * NUM_KEYS}")
    return dataset

# --- Other data generation functions would remain the same ---

# --- Feature Extraction for Neural Network ---
def extract_features_from_ciphertexts(ciphertexts, mode='raw'):
    """Extract features from ciphertexts with optional transformations"""
    # Basic raw bits extraction
    features = np.zeros((len(ciphertexts), BLOCK_SIZE*8), dtype=np.float32)
    for i, ct in enumerate(ciphertexts):
        features[i] = bytes_to_bits(ct)
    
    # Apply additional transformations based on mode
    if mode == 'raw':
        return features
    elif mode == 'byte_histograms':
        # Create byte frequency histograms (256 features per ciphertext)
        byte_histograms = np.zeros((len(ciphertexts), 256), dtype=np.float32)
        for i, ct in enumerate(ciphertexts):
            hist, _ = np.histogram(np.frombuffer(ct, dtype=np.uint8), bins=range(257), density=True)
            byte_histograms[i] = hist
        return byte_histograms
    elif mode == 'combined':
        # Combine raw bits and byte histograms
        byte_histograms = np.zeros((len(ciphertexts), 256), dtype=np.float32)
        for i, ct in enumerate(ciphertexts):
            hist, _ = np.histogram(np.frombuffer(ct, dtype=np.uint8), bins=range(257), density=True)
            byte_histograms[i] = hist
        return np.hstack((features, byte_histograms))
    else:
        raise ValueError(f"Unknown feature extraction mode: {mode}")

def bytes_to_bits(data):
    """Convert bytes to bit array"""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

# --- Enhanced Neural Network Architectures ---
def create_small_nn(input_shape, num_classes):
    """Create a small neural network (original size)"""
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
    return model

def create_medium_nn(input_shape, num_classes):
    """Create a medium-sized neural network"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
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
    return model

def create_large_nn(input_shape, num_classes):
    """Create a large neural network with residual connections"""
    inputs = keras.Input(shape=input_shape)
    
    # Initial dense layer
    x = layers.Dense(2048, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # First residual block
    skip1 = x
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(2048, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip1])
    x = layers.Activation('relu')(x)
    
    # Second residual block
    skip2 = x
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(2048, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip2])
    x = layers.Activation('relu')(x)
    
    # Transition to lower dimension
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Third residual block
    skip3 = x
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip3])
    x = layers.Activation('relu')(x)
    
    # Final layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_xlarge_nn(input_shape, num_classes):
    """Create an extra large neural network with advanced architecture"""
    inputs = keras.Input(shape=input_shape)
    
    # Split input into chunks for parallel processing
    chunk_size = input_shape[0] // 8  # Process in 8 chunks
    chunks = []
    
    for i in range(0, input_shape[0], chunk_size):
        end = min(i + chunk_size, input_shape[0])
        chunk = layers.Lambda(lambda x: x[:, i:end])(inputs)
        
        # Process this chunk
        x = layers.Dense(1024, activation='gelu')(chunk)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        
        chunks.append(x)
    
    # Combine processed chunks
    if len(chunks) > 1:
        combined = layers.Concatenate()(chunks)
    else:
        combined = chunks[0]
    
    # Main network body
    x = layers.Dense(4096, activation='gelu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # First residual block
    skip1 = x
    x = layers.Dense(4096, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(4096)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip1])
    x = layers.Activation('gelu')(x)
    
    # Second residual block
    skip2 = x
    x = layers.Dense(4096, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(4096)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip2])
    x = layers.Activation('gelu')(x)
    
    # Transition
    x = layers.Dense(2048, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Third residual block
    skip3 = x
    x = layers.Dense(2048, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(2048)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip3])
    x = layers.Activation('gelu')(x)
    
    # Final layers
    x = layers.Dense(1024, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_model(size, input_shape, num_classes):
    """Create a model based on the specified size"""
    if size == 'small':
        return create_small_nn(input_shape, num_classes)
    elif size == 'medium':
        return create_medium_nn(input_shape, num_classes)
    elif size == 'large':
        return create_large_nn(input_shape, num_classes)
    elif size == 'xlarge':
        return create_xlarge_nn(input_shape, num_classes)
    else:
        print(f"Unknown model size: {size}, falling back to medium")
        return create_medium_nn(input_shape, num_classes)

# --- Enhanced Training with Data Augmentation ---
def augment_data(X, y):
    """Augment data for training by applying bit-wise transformations"""
    augmented_X = []
    augmented_y = []
    
    # Add original data
    augmented_X.append(X)
    augmented_y.append(y)
    
    # Add bit-flipped versions (flip random bits)
    flipped_X = X.copy()
    flip_mask = np.random.random(X.shape) < 0.01  # Flip 1% of bits randomly
    flipped_X[flip_mask] = 1 - flipped_X[flip_mask]
    augmented_X.append(flipped_X)
    augmented_y.append(y)
    
    # Add bit-rotated versions
    rotated_X = np.zeros_like(X)
    rotation = 8  # Rotate by 1 byte
    for i in range(X.shape[0]):
        rotated_X[i] = np.roll(X[i], rotation)
    augmented_X.append(rotated_X)
    augmented_y.append(y)
    
    # Combine augmented data
    return np.vstack(augmented_X), np.concatenate(augmented_y)

# --- Enhanced Pattern Classifier with Cross-Validation ---
def train_large_scale_classifier(dataset, analysis_mode):
    """Train an enhanced neural network to classify ciphertexts with cross-validation"""
    print(f"\n{'='*50}")
    print(f"Training large-scale neural network classifier for {ANALYSIS_MODE}")
    print(f"{'='*50}")
    print(f"Neural network size: {NEURAL_NETWORK_SIZE}")
    print(f"Training epochs: {TRAINING_EPOCHS}")
    print(f"Data augmentation: {'Enabled' if USE_DATA_AUGMENTATION else 'Disabled'}")
    print(f"Cross-validation: {'Enabled' if USE_CROSS_VALIDATION else 'Disabled'}")
    
    if analysis_mode == "fixed_plaintext":
        # Prepare data for classification by plaintext origin
        all_features = []
        all_labels = []
        
        print("Extracting features from ciphertexts...")
        for group_idx, group in enumerate(dataset):
            # Extract features
            features = extract_features_from_ciphertexts(group['ciphertexts'], mode='raw')
            labels = np.full(len(features), group_idx)
            
            all_features.append(features)
            all_labels.append(labels)
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print(f"Feature extraction complete: {X.shape[0]} samples, {X.shape[1]} features")
        
        if USE_CROSS_VALIDATION:
            # Perform k-fold cross-validation
            k_folds = 5
            fold_size = X.shape[0] // k_folds
            accuracies = []
            
            print(f"Performing {k_folds}-fold cross-validation...")
            
            for fold in range(k_folds):
                print(f"\nTraining fold {fold+1}/{k_folds}")
                
                # Create train/test split for this fold
                test_indices = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_indices = np.concatenate([
                    np.arange(0, fold * fold_size),
                    np.arange((fold + 1) * fold_size, X.shape[0])
                ])
                
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                # Augment training data if enabled
                if USE_DATA_AUGMENTATION:
                    print("Augmenting training data...")
                    X_train, y_train = augment_data(X_train, y_train)
                    print(f"Augmented data size: {X_train.shape[0]} samples")
                
                # Create and compile model
                model = create_model(NEURAL_NETWORK_SIZE, X_train.shape[1:], NUM_PLAINTEXTS)
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Define callbacks
                callback_list = []
                if USE_EARLY_STOPPING:
                    callback_list.append(
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=PATIENCE,
                            restore_best_weights=True
                        )
                    )
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=TRAINING_EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callback_list,
                    verbose=1
                )
                
                # Evaluate
                _, accuracy = model.evaluate(X_test, y_test, verbose=1)
                accuracies.append(accuracy)
                
                # Plot training history for this fold
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Training')
                plt.plot(history.history['val_accuracy'], label='Validation')
                plt.title(f'Model Accuracy - Fold {fold+1}')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.title(f'Model Loss - Fold {fold+1}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"fold_{fold+1}_training_history.png"), dpi=300)
                plt.close()
            
            # Calculate average accuracy across folds
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            print(f"\nCross-validation complete")
            print(f"Average accuracy: {avg_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
            
            # Compare to random baseline
            random_baseline = 1.0 / NUM_PLAINTEXTS
            print(f"Random baseline: {random_baseline*100:.2f}%")
            print(f"Improvement over random: {(avg_accuracy - random_baseline)*100:.2f}%")
            
            # Train final model on all data
            print("\nTraining final model on all data...")
            
            # Augment all data if enabled
            if USE_DATA_AUGMENTATION:
                X_augmented, y_augmented = augment_data(X, y)
                print(f"Augmented full dataset size: {X_augmented.shape[0]} samples")
            else:
                X_augmented, y_augmented = X, y
            
            # Split into train/test for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_augmented, y_augmented, test_size=0.2, random_state=SEED)
            
            final_model = create_model(NEURAL_NETWORK_SIZE, X_train.shape[1:], NUM_PLAINTEXTS)
            final_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            final_history = final_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=TRAINING_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callback_list,
                verbose=1
            )
            
            # Final evaluation
            final_loss, final_accuracy = final_model.evaluate(X_test, y_test, verbose=1)
            
            print(f"\nFinal model test accuracy: {final_accuracy*100:.2f}%")
            print(f"Random baseline: {random_baseline*100:.2f}%")
            print(f"Improvement over random: {(final_accuracy - random_baseline)*100:.2f}%")
            
            # Plot final training history
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(final_history.history['accuracy'], label='Training')
            plt.plot(final_history.history['val_accuracy'], label='Validation')
            plt.title('Final Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.subplot(1, 2, 2)
            plt.plot(final_history.history['loss'], label='Training')
            plt.plot(final_history.history['val_loss'], label='Validation')
            plt.title('Final Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "final_model_training_history.png"), dpi=300)
            plt.close()
            
            # Save final model
            final_model.save(os.path.join(OUTPUT_DIR, "final_model.keras"))
            
            # Bit importance analysis
            if USE_BIT_IMPORTANCE_ANALYSIS:
                print("\nAnalyzing bit importance...")
                bit_importance = analyze_bit_importance(final_model, X_test, y_test)
                
                # Plot bit importance
                plt.figure(figsize=(14, 6))
                plt.bar(range(len(bit_importance)), bit_importance)
                plt.xlabel('Bit Position')
                plt.ylabel('Importance Score')
                plt.title('Bit Importance Analysis')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "bit_importance.png"), dpi=300)
                plt.close()
            
            return {
                'model': final_model,
                'cv_accuracy': avg_accuracy,
                'cv_std': std_accuracy,
                'final_accuracy': final_accuracy,
                'random_baseline': random_baseline,
                'history': final_history.history
            }
            
        else:
            # No cross-validation, just a single train/test split
            # Augment data if enabled
            if USE_DATA_AUGMENTATION:
                X_augmented, y_augmented = augment_data(X, y)
                print(f"Augmented dataset size: {X_augmented.shape[0]} samples")
            else:
                X_augmented, y_augmented = X, y
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_augmented, y_augmented, test_size=0.2, random_state=SEED)
            
            # Create and train model
            model = create_model(NEURAL_NETWORK_SIZE, X_train.shape[1:], NUM_PLAINTEXTS)
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Define callbacks
            callback_list = []
            if USE_EARLY_STOPPING:
                callback_list.append(
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=PATIENCE,
                        restore_best_weights=True
                    )
                )
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=TRAINING_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callback_list,
                verbose=1
            )
            
            # Evaluate
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            
            print(f"\nTest accuracy: {accuracy*100:.2f}%")
            
            # Check against random baseline
            random_baseline = 1.0 / NUM_PLAINTEXTS
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
            plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=300)
            plt.close()
            
            # Save model
            model.save(os.path.join(OUTPUT_DIR, "model.keras"))
            
            # Bit importance analysis
            if USE_BIT_IMPORTANCE_ANALYSIS:
                print("\nAnalyzing bit importance...")
                bit_importance = analyze_bit_importance(model, X_test, y_test)
                
                # Plot bit importance
                plt.figure(figsize=(14, 6))
                plt.bar(range(len(bit_importance)), bit_importance)
                plt.xlabel('Bit Position')
                plt.ylabel('Importance Score')
                plt.title('Bit Importance Analysis')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "bit_importance.png"), dpi=300)
                plt.close()
            
            return {
                'model': model,
                'accuracy': accuracy,
                'random_baseline': random_baseline,
                'history': history.history
            }
            
    else:
        print(f"Large-scale neural network analysis not implemented for {analysis_mode} mode")
        return None

def analyze_bit_importance(model, X, y, num_permutations=10):
    """
    Analyze which bits are most important for classification by permuting them
    and measuring the drop in accuracy.
    """
    # Get baseline accuracy
    baseline_loss, baseline_accuracy = model.evaluate(X, y, verbose=0)
    
    # Initialize importance scores
    bit_importance = np.zeros(X.shape[1])
    
    # Analyze each bit
    for bit_idx in tqdm(range(X.shape[1]), desc="Analyzing bits"):
        # Create multiple permutations for this bit
        accuracy_drops = []
        
        for _ in range(num_permutations):
            # Create a copy of the data
            X_permuted = X.copy()
            
            # Randomly permute this bit across samples
            X_permuted[:, bit_idx] = np.random.permutation(X_permuted[:, bit_idx])
            
            # Measure new accuracy
            loss, accuracy = model.evaluate(X_permuted, y, verbose=0)
            
            # Calculate accuracy drop
            accuracy_drop = baseline_accuracy - accuracy
            accuracy_drops.append(accuracy_drop)
        
        # Average accuracy drop across permutations
        bit_importance[bit_idx] = np.mean(accuracy_drops)
    
    return bit_importance

# --- Main Execution ---
def main():
    """Main execution function"""
    print(f"\n{'='*50}")
    print("Large-Scale AES Analysis Framework")
    print(f"{'='*50}")
    print(f"Analysis mode: {ANALYSIS_MODE}")
    print(f"Configuration: {NUM_PLAINTEXTS} plaintexts, {NUM_KEYS} keys")
    
    # Generate dataset
    dataset = generate_aes_datasets()
    
    # Train large-scale neural network
    try:
        print("\nStarting large-scale neural network training...")
        results = train_large_scale_classifier(dataset, ANALYSIS_MODE)
    except Exception as e:
        print(f"Neural network training failed: {e}")
        import traceback
        traceback.print_exc()
        results = None
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")
    return results

if __name__ == "__main__":
    results = main()

