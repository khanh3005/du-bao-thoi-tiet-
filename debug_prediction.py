"""Khi demo sai: ghi lại ảnh/model/features và so sánh hai đường đọc ảnh."""
import argparse
import numpy as np
from config import CLASSES, MODEL_PATH
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--true-label', choices=CLASSES)
    args = parser.parse_args()
    bundle = load_bundle()
    vector_file = preprocess_file(args.image)
    vector_upload = preprocess_bytes(args.image.read_bytes())
    np.testing.assert_array_equal(vector_file, vector_upload)
    p1,c1,_=predict_vectors(bundle,vector_file.reshape(1,-1))
    p2,c2,_=predict_image_bytes(bundle,args.image.read_bytes())
    np.testing.assert_array_equal(p1,p2)
    if c1 is not None: np.testing.assert_array_equal(c1,c2)
    result = {'image':str(args.image), 'image_sha256':file_hash(args.image),
              'model':bundle['name'], 'model_sha256':file_hash(MODEL_PATH), 'dataset_id':bundle['dataset_id'],
              'mapping':dict(enumerate(bundle['classes'])), 'preprocessing':bundle['preprocess'],
              'feature_shape':list(vector_file.shape), 'dtype':str(vector_file.dtype),
              'min':float(vector_file.min()), 'max':float(vector_file.max()),
              'feature_sha256':fingerprint(vector_file), 'script_equals_app_path':True,
              'predicted':CLASSES[p1[0]], 'confidence':None if c1 is None else float(c1[0]),
              'true_label':args.true_label,
              'correct':None if args.true_label is None else CLASSES[p1[0]]==args.true_label}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.true_label and not result['correct']:
        print('Hai đường inference cho cùng kết quả sai: không phải sai riêng của giao diện.')
        print('Đối chiếu báo cáo leakage, nhãn, dữ liệu và CV trước khi kết luận giới hạn model.')

if __name__ == '__main__':
    main()
