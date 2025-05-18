import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns


def scrutinize_aes_model(model_path, test_dataset, y_test, output_dir="model_analysis"):
    """
    Perform comprehensive analysis on a trained AES encryption model to verify results.

    Parameters:
    - model_path: Path to the saved model
    - test_dataset: tf.data.Dataset containing test data
    - y_test: Ground truth labels (as numpy array)
    - output_dir: Directory to save analysis outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    model.summary()

    # 1. Verify on fresh test data
    print("\n=== Model Evaluation ===")
    metrics = model.evaluate(test_dataset, verbose=1)
    metrics_names = model.metrics_names

    # Save metrics to file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        for name, value in zip(metrics_names, metrics):
            print(f"{name}: {value:.6f}")
            f.write(f"{name}: {value:.6f}\n")

    # 2. Generate predictions
    print("\n=== Generating Predictions ===")
    y_pred_probs = model.predict(test_dataset)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Ensure y_test has the right shape to match predictions
    actual_test_samples = y_pred.shape[0]
    y_test_reshaped = y_test[:actual_test_samples].reshape(y_pred.shape)

    # 3. Bit-level analysis
    print("\n=== Bit-Level Analysis ===")
    bit_accuracies = np.mean(y_pred == y_test_reshaped, axis=0)

    # 3.1 Overall bit accuracy
    bit_acc_mean = np.mean(bit_accuracies)
    bit_acc_std = np.std(bit_accuracies)
    print(f"Mean bit accuracy: {bit_acc_mean:.6f} (±{bit_acc_std:.6f})")

    # 3.2 Bits significantly above random chance (using statistical threshold)
    threshold = 0.51  # 1% above random chance
    above_threshold = np.sum(bit_accuracies > threshold)
    print(
        f"Bits above {threshold*100:.1f}% accuracy: {above_threshold}/{len(bit_accuracies)} ({above_threshold/len(bit_accuracies)*100:.2f}%)"
    )

    # 3.3 Top and bottom performing bits
    top_n = 5
    top_indices = np.argsort(bit_accuracies)[-top_n:][::-1]
    bottom_indices = np.argsort(bit_accuracies)[:top_n]

    print(f"\nTop {top_n} performing bits:")
    for i, idx in enumerate(top_indices):
        print(f"  Bit {idx}: {bit_accuracies[idx]:.6f} accuracy")

    print(f"\nBottom {top_n} performing bits:")
    for i, idx in enumerate(bottom_indices):
        print(f"  Bit {idx}: {bit_accuracies[idx]:.6f} accuracy")

    # 3.4 Plot bit accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(bit_accuracies)), bit_accuracies)
    plt.axhline(y=0.5, color="r", linestyle="--", label="Random guessing")
    plt.axhline(
        y=threshold, color="g", linestyle="--", label=f"Threshold ({threshold:.2f})"
    )
    plt.xlabel("Bit Position")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Bit Position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bit_accuracies.png"), dpi=300)

    # 4. Confusion matrix analysis (overall)
    print("\n=== Confusion Matrix Analysis ===")
    # Flatten predictions and actual values for overall analysis
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test_reshaped.flatten()

    cm = confusion_matrix(y_test_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()

    # Calculate classification metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"Overall Binary Classification Metrics:")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall: {recall:.6f}")
    print(f"  Specificity: {specificity:.6f}")
    print(f"  F1 Score: {f1:.6f}")

    # 4.1 Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix (All Bits)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)

    # 5. ROC curve analysis (sample of bits)
    print("\n=== ROC Curve Analysis ===")

    # 5.1 Overall ROC curve (using flattened data)
    fpr, tpr, _ = roc_curve(y_test_flat, y_pred_probs.flatten())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (All Bits)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_overall.png"), dpi=300)

    # 5.2 ROC curves for top 3 and bottom 3 bits
    plt.figure(figsize=(12, 8))

    # Top 3 bits
    for i, idx in enumerate(top_indices[:3]):
        fpr, tpr, _ = roc_curve(y_test_reshaped[:, idx], y_pred_probs[:, idx])
        bit_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"Bit {idx} (acc={bit_accuracies[idx]:.4f}, AUC={bit_auc:.4f})",
        )

    # Bottom 3 bits
    for i, idx in enumerate(bottom_indices[:3]):
        fpr, tpr, _ = roc_curve(y_test_reshaped[:, idx], y_pred_probs[:, idx])
        bit_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=2,
            linestyle="--",
            label=f"Bit {idx} (acc={bit_accuracies[idx]:.4f}, AUC={bit_auc:.4f})",
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Top and Bottom Performing Bits")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_bits.png"), dpi=300)

    # 6. Test with shuffled labels (sanity check)
    if hasattr(test_dataset, "unbatch"):
        print("\n=== Randomization Test ===")
        print("Running inference with shuffled labels (sanity check)...")

        # Get first batch of data
        for x_batch, _ in test_dataset.take(1):
            break

        # Create random labels
        random_y = np.random.randint(0, 2, size=y_pred_probs[: x_batch.shape[0]].shape)

        # Calculate "accuracy" against random labels
        random_acc = np.mean(y_pred[: x_batch.shape[0]] == random_y)
        print(f"Accuracy against random labels: {random_acc:.6f} (should be ~0.5)")

    # 7. Visualization of predictions vs actual for a sample
    sample_size = min(1000, actual_test_samples)
    sample_bits = min(32, y_pred.shape[1])

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(y_test_reshaped[:sample_size, :sample_bits], aspect="auto", cmap="Blues")
    plt.title("Actual Values (First Few Bits)")
    plt.ylabel("Sample")
    plt.xlabel("Bit Position")
    plt.colorbar(label="Bit Value")

    plt.subplot(2, 1, 2)
    plt.imshow(y_pred[:sample_size, :sample_bits], aspect="auto", cmap="Blues")
    plt.title("Predicted Values (First Few Bits)")
    plt.ylabel("Sample")
    plt.xlabel("Bit Position")
    plt.colorbar(label="Bit Value")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_visualization.png"), dpi=300)

    # 8. Correlation analysis between input and output bits
    print("\n=== Analysis Summary ===")
    print(f"Detailed analysis saved to: {output_dir}")
    print(f"  • Overall bit accuracy: {bit_acc_mean:.6f}")
    print(f"  • Bits above threshold: {above_threshold}/{len(bit_accuracies)}")
    print(f"  • Overall ROC AUC: {roc_auc:.6f}")

    return {
        "bit_accuracies": bit_accuracies,
        "overall_accuracy": accuracy,
        "roc_auc": roc_auc,
        "top_bits": top_indices,
        "bottom_bits": bottom_indices,
    }


def resume_training(
    model_path,
    train_dataset,
    val_dataset,
    initial_epoch,
    epochs=50,
    callbacks_list=None,
    output_dir="continued_training",
):
    """
    Resume training from a saved model

    Parameters:
    - model_path: Path to the saved model
    - train_dataset: tf.data.Dataset for training
    - val_dataset: tf.data.Dataset for validation
    - initial_epoch: Epoch to start from (0-based)
    - epochs: Total number of epochs to run
    - callbacks_list: List of callbacks to use
    - output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)

    # Add model checkpoint callback if not provided
    if callbacks_list is None:
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "model_ep{epoch:03d}.keras"),
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                os.path.join(output_dir, "training_log.csv"), append=True
            ),
        ]

    # Resume training
    print(f"Resuming training from epoch {initial_epoch}")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "continued_training_history.png"), dpi=300)

    return model, history


def export_model_for_sharing(
    model_path, output_path="exported_model", model_format="saved_model"
):
    """
    Export a trained model to a format suitable for sharing

    Parameters:
    - model_path: Path to the saved Keras model
    - output_path: Directory to save the exported model
    - model_format: Format to export ("saved_model" or "pb")
    """
    # Load the model
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)

    # Export to SavedModel format
    if model_format == "saved_model" or model_format == "both":
        saved_model_path = (
            output_path
            if model_format == "saved_model"
            else os.path.join(output_path, "saved_model")
        )
        print(f"Exporting to SavedModel format at: {saved_model_path}")
        model.save(saved_model_path, save_format="tf")

    # Export to .pb format
    if model_format == "pb" or model_format == "both":
        pb_path = (
            output_path
            if model_format == "pb"
            else os.path.join(output_path, "pb_model")
        )
        os.makedirs(pb_path, exist_ok=True)

        print(f"Exporting to .pb format at: {pb_path}")

        # Convert Keras model to concrete function
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )

        # Get frozen concrete function
        frozen_func = (
            tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(
                full_model
            )
        )
        frozen_func.graph.as_graph_def()

        # Save the model
        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=pb_path,
            name="model.pb",
            as_text=False,
        )

        # Save the weights as a separate checkpoint file
        model.save_weights(os.path.join(pb_path, "model_weights"))

        # Save model configuration
        with open(os.path.join(pb_path, "model_config.json"), "w") as f:
            f.write(model.to_json())

        print(f"Model exported to: {os.path.join(pb_path, 'model.pb')}")

    # Create README with model information
    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write("# Exported AES Neural Network Model\n\n")
        f.write("## Model Architecture\n")
        f.write("```\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("```\n\n")

        f.write("## Input Shape\n")
        f.write(f"Input shape: {model.inputs[0].shape}\n\n")

        f.write("## Output Shape\n")
        f.write(f"Output shape: {model.outputs[0].shape}\n\n")

        f.write("## Usage Example\n")
        f.write("```python\n")
        f.write("import tensorflow as tf\n\n")

        if model_format == "saved_model" or model_format == "both":
            f.write("# Load SavedModel format\n")
            f.write("model = tf.keras.models.load_model('saved_model')\n\n")

        if model_format == "pb" or model_format == "both":
            f.write("# Load .pb format\n")
            f.write("# First load the graph\n")
            f.write("with tf.io.gfile.GFile('pb_model/model.pb', 'rb') as f:\n")
            f.write("    graph_def = tf.compat.v1.GraphDef()\n")
            f.write("    graph_def.ParseFromString(f.read())\n\n")

            f.write("# Then import it into a new graph\n")
            f.write("with tf.compat.v1.Graph().as_default() as graph:\n")
            f.write("    tf.compat.v1.import_graph_def(graph_def, name='')\n\n")

            f.write("# Create a session and run inference\n")
            f.write("with tf.compat.v1.Session(graph=graph) as sess:\n")
            f.write("    # Get input and output tensors\n")
            f.write("    input_tensor = graph.get_tensor_by_name('input_1:0')\n")
            f.write("    output_tensor = graph.get_tensor_by_name('Identity:0')\n\n")

            f.write("    # Run inference\n")
            f.write(
                "    output = sess.run(output_tensor, {input_tensor: input_data})\n"
            )

        f.write("```\n")

    print(
        f"Model export complete. Documentation saved to: {os.path.join(output_path, 'README.md')}"
    )

    return output_path


# Example usage:
if __name__ == "__main__":
    # These functions would be called from your main script, with appropriate parameters
    print("Import this module to use the functions")

    # Example calls (commented out):
    """
    # 1. Scrutinize a trained model
    scrutinize_aes_model(
        model_path="aes_nn_models/aes_cnn_model_best.keras",
        test_dataset=test_dataset,
        y_test=y_test,
        output_dir="model_analysis_cnn"
    )
    
    # 2. Resume training from a previous checkpoint
    resume_training(
        model_path="aes_nn_models/aes_transformer_model_e045_vl0.1090.keras",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        initial_epoch=45,
        epochs=100
    )
    
    # 3. Export model for sharing
    export_model_for_sharing(
        model_path="aes_nn_models/aes_cnn_model_best.keras",
        output_path="exported_cnn_model",
        model_format="both"  # Save both SavedModel and .pb formats
    )
    """
