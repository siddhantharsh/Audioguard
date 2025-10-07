import sounddevice as sd
import numpy as np
import queue
import time
import tensorflow as tf
from preprocess import compute_log_mel, load_audio
import config as cfg
from collections import deque

MODEL_PATH = cfg.MODELS_DIR / "sound_model.keras"
WINDOW_SECONDS = cfg.DURATION
SR = cfg.SAMPLE_RATE
CHUNK = 1024

# load model
model = tf.keras.models.load_model(MODEL_PATH)
labels = cfg.CLASS_NAMES

q = queue.Queue()
buffer = np.zeros(int(SR * WINDOW_SECONDS), dtype=np.float32)
smooth = deque(maxlen=3)  # smoothing last 3 confidences

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def infer(buffer):
    spec = compute_log_mel(buffer)
    # model expects (1, time, n_mels)
    x = np.expand_dims(spec, axis=0).astype(np.float32)
    probs = model.predict(x, verbose=0)[0]
    return probs

def run():
    # stream
    stream = sd.InputStream(samplerate=SR, channels=1, blocksize=CHUNK, callback=callback)
    stream.start()
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            # fill buffer with rolling audio
            while not q.empty():
                data = q.get()
                data = data.reshape(-1)
                buffer[:-len(data)] = buffer[len(data):]
                buffer[-len(data):] = data
            probs = infer(buffer)
            smooth.append(probs)
            avg = np.mean(smooth, axis=0)
            idx = int(np.argmax(avg))
            print(f"\r{labels[idx]} ({avg[idx]*100:.1f}%)", end="", flush=True)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    run()
