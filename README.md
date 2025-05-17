# Initial Report: Neural Network for AES Encryption Learning

**Date:** May 17, 2025

## 1. Goal

The primary objective of this experiment is to develop and train a neural network using TensorFlow and Python to learn the Advanced Encryption Standard (AES) encryption function. This involved:
1.  Generating a dataset of plaintext, key, and corresponding ciphertext tuples.
2.  Defining multiple neural network model (Multi-Layer Perceptron, Transformer, Hybrid, etc).
3.  Experimenting with different optimizations.
4.  Training the model on the generated dataset.
5.  Evaluating the model's performance in predicting ciphertexts.
6.  Utilizing CUDA and a NVIDIA 3070 GPU for acceleration for efficient training.

## 2. Process and Walkthrough

The experiment has progressed through several stages, with a significant focus on environment setup and troubleshooting GPU compatibility for TensorFlow. We begin with a simple MLP setup and a small data set.

### Data Generation:
```python
def generate_sample():
key = np.random.bytes(16)
plaintext = np.random.bytes(16)
cipher = AES.new(key, AES.MODE_ECB)
ciphertext = cipher.encrypt(plaintext)
return np.unpackbits(np.frombuffer(plaintext, dtype=np.uint8)),
np.unpackbits(np.frombuffer(key, dtype=np.uint8)),
np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
```
### Model:

```python
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
```
![alt text](image.png)

1. Initial model was a Keras Sequential with 2 hidden layers (1024 units each, ReLU), a smaller hidden layer (512 units, ReLU), and an output layer (128 units, sigmoid). Also added Batch Normalization and Dropout. 
2. For training, I used the Adam optimizer and binary cross-entropy loss, also stopping early if enough epochs went by without a drop in validation loss.
3. Concerning reproducibility, random seeds were set for numpy and tensorflow. 

### It should be noted that results of any consequence were not expected with a dataset this small, using a simple MLP setup. 

## Initial Results

![alt text](training_history.png)

-