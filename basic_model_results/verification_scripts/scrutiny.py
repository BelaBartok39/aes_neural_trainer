from aes_dataset_generator import generate_fresh_aes_dataset, verify_dataset_integrity
from aes_model_tools import scrutinize_aes_model

# Generate a fresh dataset
print("Generating fresh AES dataset...")
dataset = generate_fresh_aes_dataset(
    num_samples=100000,
    bits_to_predict=32,
    seed=42,
    output_dir="fresh_aes_dataset"
)

# Verify integrity (no data leakage)
verify_dataset_integrity("fresh_aes_dataset")

# Extract the components needed for scrutinize_aes_model
test_dataset = dataset['test_dataset']
y_test = dataset['y_test']

# Analyze your trained model with the fresh dataset
model_path = "aes_nn_cnn_final_model.keras"  # Change this to your model path
print(f"Analyzing model {model_path} with fresh test data...")
results = scrutinize_aes_model(
    model_path=model_path,
    test_dataset=test_dataset,
    y_test=y_test,
    output_dir="fresh_model_analysis"
)

# Print key findings
print(f"Overall bit accuracy: {results['overall_accuracy']:.4f}")
print(f"Best performing bit: {results['top_bits'][0]} with accuracy {results['bit_accuracies'][results['top_bits'][0]]:.4f}")
print(f"Worst performing bit: {results['bottom_bits'][0]} with accuracy {results['bit_accuracies'][results['bottom_bits'][0]]:.4f}")
