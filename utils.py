"""Một dây chuyền chung: đọc bytes -> BGR -> RGB -> resize -> vector."""
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version
import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from config import CLASSES, IMAGE_SIZE, FEATURE_COUNT, PREPROCESS_SPEC, MODEL_PATH, PROCESSED, N_JOBS


def is_image_file(path):
    return Path(path).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def decode_image(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không đọc được ảnh; kiểm tra nội dung file.')
    return image


def preprocess_bgr(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Đầu vào phải là ảnh BGR uint8 có 3 kênh.')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    vector = (rgb.astype(np.float32) / np.float32(255)).flatten(order='C')
    assert vector.shape == (FEATURE_COUNT,) and np.isfinite(vector).all()
    return vector


def preprocess_bytes(data):
    return preprocess_bgr(decode_image(data))


def preprocess_file(path):
    return preprocess_bytes(Path(path).read_bytes())


def fingerprint(vector):
    return hashlib.sha256(np.asarray(vector, dtype='<f4').tobytes(order='C')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(vector):
    # Hash thị giác: nhận diện ứng viên ảnh gần trùng; KHÔNG chứng minh cùng ảnh.
    rgb = (vector.reshape(64, 64, 3) * 255).round().astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    values = cv2.dct(small)[:8, :8].flatten()[1:]
    bits = values > np.median(values)
    return sum(int(v) << i for i, v in enumerate(bits))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def environment():
    names = ['numpy', 'pandas', 'scikit-learn', 'opencv-python-headless',
             'xgboost', 'lightgbm', 'streamlit', 'joblib', 'scipy', 'matplotlib']
    return {'python': platform.python_version(), **{n: version(n) for n in names}}


def load_bundle(path=MODEL_PATH):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get('schema') != 1:
        raise ValueError('Model cũ/sai định dạng. Chạy lại train_models.py.')
    if bundle['classes'] != list(CLASSES) or bundle['preprocess'] != PREPROCESS_SPEC:
        raise ValueError('Mapping hoặc preprocessing không khớp model.')
    model = bundle['model']
    if list(model.classes_) != list(range(len(CLASSES))):
        raise ValueError('Thứ tự model.classes_ không khớp mapping.')
    if model.n_features_in_ != FEATURE_COUNT:
        raise ValueError('Model không nhận 12.288 features.')
    # Chặn model chạy bằng phiên bản sklearn khác, tránh kết quả không đáng tin.
    if bundle['environment']['scikit-learn'] != version('scikit-learn'):
        raise ValueError('Phiên bản scikit-learn khác lúc train. Cài requirements.txt hoặc train lại.')
    metadata_path = PROCESSED / 'metadata.json'
    if not metadata_path.exists():
        raise ValueError('Thiếu metadata preprocessing thành công; không nạp model có thể đã cũ.')
    processed_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if processed_metadata['dataset_id'] != bundle['dataset_id']:
        raise ValueError('Model thuộc dữ liệu cũ; dataset_id không khớp processed.')
    return bundle


def predict_vectors(bundle, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT or not np.isfinite(X).all():
        raise ValueError('Features sai shape hoặc có NaN/Inf.')
    if X.min() < 0 or X.max() > 1:
        raise ValueError('Features phải trong [0, 1].')
    model = bundle['model']
    # Giới hạn BLAS: nhiều luồng cho một ảnh nhỏ có thể chậm hơn đáng kể.
    with threadpool_limits(limits=N_JOBS):
        predictions = model.predict(X).astype(int)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    # Lấy xác suất CỦA NHÃN predict, không lấy max rồi gắn nhầm nhãn của SVC.
    confidence = None if probabilities is None else np.array([
        probabilities[i, list(model.classes_).index(int(label))]
        for i, label in enumerate(predictions)])
    return predictions, confidence, probabilities


def predict_image_bytes(bundle, data):
    return predict_vectors(bundle, preprocess_bytes(data).reshape(1, -1))
