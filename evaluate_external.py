"""External độc lập. Nhãn khai báo rõ trong external_labels.csv, không tự đoán."""
import numpy as np
import pandas as pd
from config import *
from utils import *
from evaluation import save_evaluation


def main():
    bundle = load_bundle()
    clean = pd.read_csv(REPORTS/'clean_manifest.csv', dtype={'phash':str})
    known = set(clean['hash'])
    hashes = [int(v) for v in clean.phash]
    labels_path = ROOT / 'external_labels.csv'
    label_rows = pd.read_csv(labels_path).fillna('') if labels_path.exists() else pd.DataFrame(columns=['file','label'])
    if label_rows['file'].duplicated().any():
        raise ValueError('external_labels.csv có tên file lặp.')
    if not set(label_rows.label).issubset(set(CLASSES) | {''}):
        raise ValueError('Nhãn external không hợp lệ.')
    labels = dict(zip(label_rows.file, label_rows.label))
    rows = []
    for path in sorted(EXTERNAL.rglob('*')):
        if not path.is_file() or not is_image_file(path):
            continue
        relative = path.relative_to(EXTERNAL).as_posix()
        row = {'file':relative, 'true_label':labels.get(relative,''), 'predicted':'',
               'confidence':np.nan, 'correct':'', 'excluded_reason':''}
        try:
            vector = preprocess_file(path)
            phash = perceptual_hash(vector)
            if fingerprint(vector) in known:
                row['excluded_reason'] = 'exact_duplicate_of_dataset'
            elif any((phash ^ h).bit_count() <= PHASH_DISTANCE for h in hashes):
                row['excluded_reason'] = 'near_duplicate_candidate_of_dataset'
            prediction, confidence, _ = predict_vectors(bundle,vector.reshape(1,-1))
            row['predicted'] = CLASSES[prediction[0]]
            row['confidence'] = confidence[0] if confidence is not None else np.nan
            if row['true_label']:
                row['correct'] = row['true_label'] == row['predicted']
        except Exception as e:
            row['excluded_reason'] = f'error: {e}'
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('Không tìm thấy ảnh external.')
    frame.to_csv(REPORTS/'external_predictions.csv',index=False)
    errors = frame[frame.excluded_reason.str.startswith('error:')]
    if not errors.empty:
        raise RuntimeError('External có lỗi đọc ảnh/dự đoán; xem external_predictions.csv. Chưa đánh giá thành công.')
    valid = frame[(frame.true_label != '') & (frame.excluded_reason == '')]
    if not valid.empty:
        metrics = save_evaluation([CLASSES.index(v) for v in valid.true_label],
                                  [CLASSES.index(v) for v in valid.predicted], 'external')
        print(metrics)
    print(frame.to_string(index=False))
    print('Nhãn external do người dùng cung cấp/xác nhận. Không dùng điểm external để chọn tham số.')

if __name__ == '__main__':
    main()
