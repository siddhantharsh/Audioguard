import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer
import config as cfg
from dataset import build_file_list, create_tf_dataset
from model import build_transformer_audio_model
from tensorflow.keras import callbacks, optimizers, losses

# --- build file lists
paths, labels, le = build_file_list(cfg.METADATA_CSV)
cfg.CLASS_NAMES = list(le.classes_)
num_classes = len(cfg.CLASS_NAMES)
print("Classes:", cfg.CLASS_NAMES)

# --- train/val/test split (stratified)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Further split train into train and validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Create datasets
train_ds = create_tf_dataset(train_paths, train_labels, augment=True, shuffle=True)
val_ds = create_tf_dataset(val_paths, val_labels, augment=False, shuffle=False)

# Create artifacts directories
os.makedirs(os.path.join("artifacts", "data"), exist_ok=True)
os.makedirs(os.path.join("artifacts", "models"), exist_ok=True)

# Save test data for evaluation
X_test, y_test = create_tf_dataset(test_paths, test_labels, augment=False, shuffle=False, return_numpy=True)
np.save(os.path.join("artifacts", "data", "X_test.npy"), X_test)
np.save(os.path.join("artifacts", "data", "y_test.npy"), y_test)

# class weights
class_weights_raw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights_raw))

# --- model
model = build_transformer_audio_model(num_classes)
opt = optimizers.Adam(learning_rate=cfg.LR)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)
model.summary()

# callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
ckp = callbacks.ModelCheckpoint(str(cfg.MODELS_DIR / "sound_model.keras"), save_best_only=True, monitor="val_loss")
tensorboard_cb = callbacks.TensorBoard(log_dir="logs")

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=cfg.EPOCHS,
                    callbacks=[es, rlr, ckp, tensorboard_cb],
                    class_weight=class_weights)

# Save training history
import json
history_dict = history.history
with open(os.path.join("artifacts", "models", "training_history.json"), 'w') as f:
    json.dump(history_dict, f)

# save final model
model.save(cfg.MODELS_DIR / "final_sound_model.keras")
