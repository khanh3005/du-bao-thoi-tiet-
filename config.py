"""Cấu hình duy nhất cho dữ liệu, huấn luyện và app."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'dataset'
EXTERNAL = ROOT / 'alien_test'
PROCESSED = ROOT / 'processed'
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models'
MODEL_PATH = MODELS / 'best_weather_model.pkl'
CLASSES = ('cloudy', 'foggy', 'rainy', 'shine', 'sunrise')
IMAGE_SIZE = (64, 64)
FEATURE_COUNT = 12288
SEED = 42
CV_FOLDS = 3
N_JOBS = 2
PHASH_DISTANCE = 6
PREPROCESS_SPEC = {'size': [64, 64], 'color': 'RGB', 'dtype': 'float32',
                   'scale': 255, 'order': 'C', 'resize': 'INTER_LINEAR',
                   'decoder': 'cv2.IMREAD_COLOR', 'features': FEATURE_COUNT}

def make_dirs():
    for p in (PROCESSED, REPORTS, MODELS):
        p.mkdir(parents=True, exist_ok=True)
