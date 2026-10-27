"""Dự đoán độc lập app: 3 ảnh/class từ Train; không phải đánh giá generalization."""
import argparse
import numpy as np
import pandas as pd
from config import *
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, help='Dự đoán một file ảnh bất kỳ')
    args = parser.parse_args()
    bundle = load_bundle()
    if args.image:
        pred, confidence, _ = predict_image_bytes(bundle,args.image.read_bytes())
        print(args.image, CLASSES[pred[0]], 'confidence:', None if confidence is None else confidence[0])
        return
    manifest = pd.read_csv(REPORTS/'train_manifest.csv')
    saved = np.load(PROCESSED/'X_train.npy',allow_pickle=False)
    rows=[]
    for label in range(5):
        indices = manifest.index[manifest.label == label][:3]
        for index in indices:
            row=manifest.loc[index]
            data=(ROOT/row.path).read_bytes()
            vector=preprocess_bytes(data)
            assert np.array_equal(vector,saved[index]), 'Train/upload preprocessing khác nhau!'
            p,c,_=predict_image_bytes(bundle,data)
            rows.append({'file':row.path,'true_label':CLASSES[label], 'predicted':CLASSES[p[0]],
                         'confidence':c[0] if c is not None else np.nan, 'correct':int(p[0])==label})
    frame=pd.DataFrame(rows)
    frame.to_csv(REPORTS/'sanity_predictions.csv',index=False)
    print(frame.to_string(index=False))
    print('Preprocess file/bytes/saved features: IDENTICAL. Đây là ảnh Train, không phải điểm Test.')

if __name__ == '__main__':
    main()
