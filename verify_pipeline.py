"""Kiểm tra tính đúng của pipeline; không điều chỉnh model bằng kết quả Test."""
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import *
from utils import *


def main():
    checks = []
    # Màu BGR [10,20,30] phải thành RGB [30,20,10], C order.
    bgr = np.zeros((64,64,3),dtype=np.uint8)
    bgr[:] = [10,20,30]
    vector=preprocess_bgr(bgr)
    assert vector.dtype==np.float32 and vector.shape==(12288,)
    np.testing.assert_array_equal(vector[:3],np.array([30,20,10],dtype=np.float32)/255)
    bgr[0,1]=[40,50,60]
    np.testing.assert_array_equal(preprocess_bgr(bgr)[3:6],np.array([60,50,40],dtype=np.float32)/255)
    ok, encoded=cv2.imencode('.png',bgr)
    assert ok
    np.testing.assert_array_equal(preprocess_bytes(encoded.tobytes()),preprocess_bgr(bgr))
    checks.append('RGB, /255, float32, C-order, lossless upload: PASS')
    try:
        preprocess_bytes(b'not an image')
        raise AssertionError('Ảnh hỏng không bị chặn')
    except ValueError:
        pass
    checks.append('Corrupt input rejected: PASS')
    assert np.load(PROCESSED/'classes.npy').tolist()==list(CLASSES)
    meta=json.loads((PROCESSED/'metadata.json').read_text())
    for path,digest in meta['checksums'].items():
        assert file_hash(ROOT/path)==digest,path
    hashes,groups={},{}
    for split in ['train','test']:
        X=np.load(PROCESSED/f'X_{split}.npy')
        y=np.load(PROCESSED/f'y_{split}.npy')
        g=np.load(PROCESSED/f'groups_{split}.npy')
        m=pd.read_csv(REPORTS/f'{split}_manifest.csv')
        assert X.shape==(len(m),FEATURE_COUNT) and X.dtype==np.float32
        assert np.array_equal(y,m.label) and np.array_equal(g,m.group)
        for i,row in m.iterrows():
            v=preprocess_file(ROOT/row.path)
            assert np.array_equal(v,X[i]),row.path
            assert fingerprint(v)==row['hash']
            assert CLASSES[y[i]]==row['class']
        hashes[split]=set(m['hash']);groups[split]=set(g)
        if split=='train':
            for a,b in StratifiedGroupKFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED).split(X,y,g):
                assert not set(g[a])&set(g[b])
    assert not hashes['train']&hashes['test']
    assert not groups['train']&groups['test']
    checks.append('All saved features equal inference preprocessing; label/manifest alignment: PASS')
    checks.append('Train/Test exact and detected near-duplicate group overlap = 0; CV group overlap = 0: PASS')
    if MODEL_PATH.exists():
        bundle=load_bundle()
        assert bundle['dataset_id']==meta['dataset_id']
        metadata=json.loads((MODELS/'model_metadata.json').read_text())
        assert file_hash(MODEL_PATH)==metadata['model_sha256']
        m=pd.read_csv(REPORTS/'train_manifest.csv')
        for path in m.groupby('class',sort=True).head(2).path:
            data=(ROOT/path).read_bytes()
            a=predict_image_bytes(bundle,data)
            b=predict_vectors(bundle,preprocess_file(ROOT/path).reshape(1,-1))
            for av,bv in zip(a,b):
                if av is not None: np.testing.assert_array_equal(av,bv)
        checks.append('Serialized model metadata, mapping, digest, and file/upload predictions: PASS')
    write_json(REPORTS/'pipeline_checks.json', {'checks':checks,'model_checked':MODEL_PATH.exists()})
    print('\n'.join(checks))

if __name__ == '__main__':
    main()
