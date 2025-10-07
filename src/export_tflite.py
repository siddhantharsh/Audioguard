import tensorflow as tf
import config as cfg

model = tf.keras.models.load_model(cfg.MODELS_DIR / "sound_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# if you have calibration data, set converter.representative_dataset
tflite_model = converter.convert()
open(cfg.MODELS_DIR / "sound_model.tflite", "wb").write(tflite_model)
print("Saved TFLite model.")