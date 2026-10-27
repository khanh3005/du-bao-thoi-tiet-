"""Khử trùng chính xác, nhóm ảnh gần trùng, rồi mới chia dữ liệu."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import *
from utils import *
def main():
    make_dirs()
    marker = PROCESSED / 'metadata.json'
    # Khi đã đánh giá Test, giữ nguyên dữ liệu; không vô tình tạo đề thi mới.
    if (REPORTS / 'final_test_metrics.json').exists() or (REPORTS / 'final_test_started.json').exists():
        raise RuntimeError('Run này đã đánh giá Test. Giữ nguyên processed; xem README để tạo nghiên cứu mới.')
    marker.unlink(missing_ok=True)
    if sorted(p.name for p in DATASET.iterdir() if p.is_dir()) != list(CLASSES):
        raise ValueError('Tên folder dataset không khớp 5 class.')
    rows, vectors, duplicates, conflicts, invalid = [], [], [], [], []
    seen, raw_counts = {}, Counter()
    for label, name in enumerate(CLASSES):
        for path in sorted((DATASET / name).rglob('*')):
            if not path.is_file() or not is_image_file(path):
                continue
            raw_counts[name] += 1
            relative = path.relative_to(ROOT).as_posix()
            try:
                vector = preprocess_file(path)
            except Exception as e:
                invalid.append({'path': relative, 'error': str(e)})
                continue
            key = fingerprint(vector)
            if key in seen:
                original = rows[seen[key]]
                item = {'path': relative, 'kept_path': original['path'],
                        'label': name, 'kept_label': original['class'], 'hash': key}
                duplicates.append(item)
                if original['label'] != label:
                    conflicts.append(item)
                continue
            seen[key] = len(rows)
            rows.append({'path': relative, 'label': label, 'class': name,
                         'hash': key, 'phash': str(perceptual_hash(vector))})
            vectors.append(vector)
    for name, data, cols in [
        ('duplicates',duplicates,['path','kept_path','label','kept_label','hash']),
        ('label_conflicts',conflicts,['path','kept_path','label','kept_label','hash']),
        ('invalid_images',invalid,['path','error'])]:
        pd.DataFrame(data, columns=cols).to_csv(REPORTS / f'{name}.csv', index=False)
    if conflicts or invalid:
        raise ValueError('Có nhãn mâu thuẫn hoặc ảnh hỏng. Xem reports và sửa dữ liệu trước khi tiếp tục.')
    if not vectors:
        raise ValueError('Không có ảnh hợp lệ.')
    X = np.stack(vectors)
    manifest = pd.DataFrame(rows)
    y = manifest.label.to_numpy(dtype=np.int64)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    near = []
    hashes = [int(r['phash']) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= PHASH_DISTANCE:
                parent[find(i)] = find(j)
                near.append({'path_a': rows[j]['path'], 'path_b': rows[i]['path'],
                             'distance': distance, 'class_a': rows[j]['class'], 'class_b': rows[i]['class']})
    groups = np.array([find(i) for i in range(len(rows))])
    manifest['group'] = groups
    pd.DataFrame(near, columns=['path_a','path_b','distance','class_a','class_b']).to_csv(REPORTS / 'near_duplicates.csv', index=False)
    # ~20% Test. Một nhóm ảnh gần trùng chỉ nằm ở một phía.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(splitter.split(X, y, groups))
    overlap = set(manifest.iloc[train]['hash']) & set(manifest.iloc[test]['hash'])
    group_overlap = set(groups[train]) & set(groups[test])
    assert not overlap and not group_overlap
    for indices in (train, test):
        assert set(y[indices]) == set(range(5)), 'Một tập thiếu class.'
    for label in range(5):
        assert len(set(groups[train][y[train] == label])) >= CV_FOLDS
    for split, indices in [('train', train), ('test', test)]:
        np.save(PROCESSED / f'X_{split}.npy', X[indices])
        np.save(PROCESSED / f'y_{split}.npy', y[indices])
        np.save(PROCESSED / f'groups_{split}.npy', groups[indices])
        manifest.iloc[indices].to_csv(REPORTS / f'{split}_manifest.csv', index=False)
    np.save(PROCESSED / 'classes.npy', np.array(CLASSES))
    manifest.to_csv(REPORTS / 'clean_manifest.csv', index=False)
    summary = [{'class': name, 'raw': raw_counts[name], 'unique': int(sum(y == i)),
                'train': int(sum(y[train] == i)), 'test': int(sum(y[test] == i))}
               for i, name in enumerate(CLASSES)]
    pd.DataFrame(summary).to_csv(REPORTS / 'dataset_summary.csv', index=False)
    artifacts = [*PROCESSED.glob('*.npy'), *(REPORTS / f'{s}_manifest.csv' for s in ['clean','train','test'])]
    checksums = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(artifacts)}
    dataset_id = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    meta = {'dataset_id': dataset_id, 'classes': list(CLASSES), 'preprocess': PREPROCESS_SPEC,
            'raw': sum(raw_counts.values()), 'unique': len(y), 'removed_duplicates': len(duplicates),
            'near_duplicate_pairs': len(near), 'phash_threshold': PHASH_DISTANCE,
            'train_count': len(train), 'test_count': len(test), 'exact_overlap': 0, 'group_overlap': 0,
            'split': 'First fold of StratifiedGroupKFold(5, shuffle=True, random_state=42)',
            'checksums': checksums}
    write_json(marker, meta)
    print(pd.DataFrame(summary).to_string(index=False))
    print('Train/Test duplicate overlap: 0; near-duplicate group overlap: 0')
    print(json.dumps({k:v for k,v in meta.items() if k!='checksums'}, indent=2))

if __name__ == '__main__':
    main()
