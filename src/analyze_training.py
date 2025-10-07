import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_training_history(history_path):
    """Load training history from JSON file."""
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history file not found at {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def plot_training_history(history):
    """Plot training and validation metrics."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot training & validation accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('artifacts/charts/training_history.png')
    plt.close()

def analyze_overfitting(history):
    """Analyze training history for overfitting indicators."""
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    # Calculate final metrics
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    
    # Calculate accuracy gap
    acc_gap = final_train_acc - final_val_acc
    
    # Find best validation accuracy and corresponding epoch
    best_val_acc_epoch = np.argmax(val_acc)
    best_val_acc = val_acc[best_val_acc_epoch]
    
    # Check if validation metrics are getting worse
    val_loss_trend = val_loss[-5:]  # Look at last 5 epochs
    is_val_loss_increasing = all(val_loss_trend[i] > val_loss_trend[i-1] for i in range(1, len(val_loss_trend)))
    
    print("\nOverfitting Analysis:")
    print("-" * 50)
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Accuracy Gap: {acc_gap:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch+1})")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Provide overfitting assessment
    print("\nOverfitting Assessment:")
    if acc_gap > 0.1:  # More than 10% gap
        print("⚠️ Large gap between training and validation accuracy suggests overfitting")
    if is_val_loss_increasing:
        print("⚠️ Validation loss is consistently increasing in the last 5 epochs (sign of overfitting)")
    if best_val_acc_epoch < len(val_acc) - 10:
        print(f"⚠️ Best validation accuracy was achieved {len(val_acc) - best_val_acc_epoch} epochs ago")
    
    # Recommendations
    print("\nRecommendations:")
    if acc_gap > 0.1:
        print("- Consider adding dropout layers or increasing dropout rate")
        print("- Add L1/L2 regularization to the model")
        print("- Reduce model complexity")
    if is_val_loss_increasing:
        print("- Implement early stopping")
        print("- Reduce learning rate")
    if acc_gap <= 0.1 and not is_val_loss_increasing:
        print("✅ Model shows no significant signs of overfitting")
        if final_val_acc < 0.9:
            print("- Consider training for more epochs or increasing model capacity")

if __name__ == "__main__":
    history_path = "artifacts/models/training_history.json"
    try:
        history = load_training_history(history_path)
        plot_training_history(history)
        analyze_overfitting(history)
    except FileNotFoundError:
        print(f"Error: Training history file not found at {history_path}")
        print("Make sure to train the model first and save the history")