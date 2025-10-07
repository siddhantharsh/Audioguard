import os
import numpy as np
import tensorflow as tf
from model import Patchify  # 👈 import custom layer so Keras can see it

MODEL_PATH = os.path.join("artifacts", "models", "sound_model.keras")
X_TEST_PATH = os.path.join("artifacts", "data", "X_test.npy")
Y_TEST_PATH = os.path.join("artifacts", "data", "y_test.npy")

# Load test data
X_test = np.load(X_TEST_PATH, allow_pickle=True)
y_test = np.load(Y_TEST_PATH, allow_pickle=True)

# Load model with custom_objects (safe way)
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"Patchify": Patchify}
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"✅ Evaluation complete - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
