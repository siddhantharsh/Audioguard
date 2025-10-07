import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from config import METADATA_CSV, DATA_DIR, SAMPLE_RATE, DURATION, BATCH_SIZE, CLASS_NAMES
from preprocess import load_audio, compute_log_mel, spec_augment
from sklearn.preprocessing import LabelEncoder

def build_file_list(metadata_csv=METADATA_CSV):
    df = pd.read_csv(metadata_csv)
    # adjust columns to your CSV structure; UrbanSound8K uses 'fold' and 'slice_file_name'
    paths = []
    labels = []
    for _, r in df.iterrows():
        fold = int(r["fold"])
        fname = r["slice_file_name"]
        label = r["class"]
        p = DATA_DIR / "audio" / f"fold{fold}" / fname
        if p.exists():
            paths.append(str(p))
            labels.append(label)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # update class names in config (side-effect)
    from config import CLASS_NAMES as _cn; 
    if _cn is None:
        import config as cfg
        cfg.CLASS_NAMES = list(le.classes_)
    return np.array(paths), y, le

def _py_loader(path, label, augment=False):
    path = path.numpy().decode("utf-8") if isinstance(path, tf.Tensor) else path
    y = load_audio(path)
    spec = compute_log_mel(y)
    if augment:
        spec = spec_augment(spec)
    # shape: (time_steps, n_mels)
    return spec.astype(np.float32), np.int32(label)

def create_tf_dataset(paths, labels, augment=False, shuffle=True, return_numpy=False):
    if return_numpy:
        # Process all files and return numpy arrays
        specs = []
        for path in paths:
            spec, _ = _py_loader(path, 0, augment=augment)  # label doesn't matter here
            specs.append(spec)
        # Pad to max length
        max_len = max(spec.shape[0] for spec in specs)
        padded_specs = []
        for spec in specs:
            pad_len = max_len - spec.shape[0]
            padded_spec = np.pad(spec, ((0, pad_len), (0, 0)), mode='constant')
            padded_specs.append(padded_spec)
        return np.array(padded_specs), np.array(labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    # map with py_function to do librosa preprocessing
    def _load_py(path, label):
        spec, lab = tf.py_function(func=_py_loader, inp=[path, label, augment], Tout=[tf.float32, tf.int32])
        spec.set_shape([None, None])  # dynamic time dimension
        lab.set_shape([])
        return spec, lab

    dataset = dataset.map(_load_py, num_parallel_calls=tf.data.AUTOTUNE)

    # pad sequences to same time_steps within batch (use padding)
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=([None, None], []), drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
