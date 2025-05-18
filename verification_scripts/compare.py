from aes_model_comparison import compare_model_with_aes

results = compare_model_with_aes(
    model_path="aes_nn_models/aes_cnn_model_e99_vl0.0070.keras", 
    num_samples=20,              # Number of random samples to test
    bits_to_predict=32,          # Same as in your training
    visualize=True,              # Create comparison visualizations
    output_dir="aes_comparisons" # Where to save results
)

# Access detailed statistics
overall_accuracy = results['overall_accuracy']  # Should be close to your test accuracy
bit_position_accuracy = results['bit_position_accuracy']  # Accuracy per bit position
best_bit = results['best_bit']  # Which bit position is most predictable
