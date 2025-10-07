from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
REPORTS_DIR = ARTIFACTS / "reports"
CHARTS_DIR = ARTIFACTS / "charts"
DATA_DIR = ROOT / "data" / "UrbanSound8K"
METADATA_CSV = DATA_DIR / "metadata" / "UrbanSound8K.csv"

SAMPLE_RATE = 22050
DURATION = 4.0   # seconds used for each sample
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048
PATCH_SIZE = 8   # time frames per patch
DMODEL = 256
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
CLASS_NAMES = None   # set dynamically after loading metadata

# ensure folders exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
