import numpy as np
import librosa
from config import SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH, N_FFT
import random

SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode="constant")
    else:
        y = y[:SAMPLES_PER_TRACK]
    return y

def compute_log_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize per-sample to zero mean unit var (stabilizes training)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)
    return log_mel.T  # transpose to shape (time_steps, n_mels)

# --- SpecAugment ---
def spec_augment(spec, freq_mask_max=10, time_mask_max=20, num_freq_masks=2, num_time_masks=2):
    spec = spec.copy()
    n_time, n_mel = spec.shape
    # frequency masks
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_max)
        f0 = random.randint(0, max(0, n_mel - f))
        spec[:, f0:f0+f] = 0
    # time masks
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_max)
        t0 = random.randint(0, max(0, n_time - t))
        spec[t0:t0+t, :] = 0
    return spec

# --- Mixup helper (outside tf)
def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
