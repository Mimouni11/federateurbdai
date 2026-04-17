from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLEGAN_ROOT = Path(__file__).resolve().parent

STYLEGAN_DIR = PROJECT_ROOT / "vendor" / "stylegan2-ada-pytorch"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "ffhq512.pkl"
OUTPUT_DIR = STYLEGAN_ROOT / "outputs"
INPUT_DIR = STYLEGAN_ROOT / "inputs"
CELEBRITY_DB_DIR = STYLEGAN_ROOT / "celebrity_db"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

CLIP_MODEL = "ViT-B/32"
CLIP_NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]

NUM_STEPS = 300
LEARNING_RATE = 0.05
TRUNCATION_PSI = 0.7
W_MEAN_SAMPLES = 1000
USE_FP16 = False