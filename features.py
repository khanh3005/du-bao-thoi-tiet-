"""Đặc trưng thủ công thử nghiệm: màu, tương phản theo vùng và HOG.
Input vẫn là vector RGB float32 64×64×3 từ utils.py, không dùng Deep Learning.
"""
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from config import FEATURE_COUNT


class WeatherFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != FEATURE_COUNT:
            raise ValueError('WeatherFeatures cần 12.288 raw RGB features.')
        self.n_features_in_ = FEATURE_COUNT
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Sai số features.')
        hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
        rows=[]
        for vector in X:
            rgb = np.rint(np.clip(vector,0,1).reshape(64,64,3)*255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv /= np.array([180,255,255],dtype=np.float32)
            gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            parts=[]
            # Histogram mô tả lượng màu; không buộc mây ở cùng tọa độ.
            for channel in range(3):
                hist=np.histogram(hsv[:,:,channel],bins=16,range=(0,1))[0].astype(np.float32)
                parts.append(hist / hist.sum())
            # Mean/std từng vùng 16×16: độ sáng và độ đồng đều (gợi ý sương).
            for y in range(0,64,16):
                for x in range(0,64,16):
                    block=hsv[y:y+16,x:x+16]
                    parts += [block.mean(axis=(0,1)),block.std(axis=(0,1))]
            # HOG mô tả hướng biến thiên sáng/tối, không phải mạng neural.
            parts.append(hog.compute(gray).ravel())
            rows.append(np.concatenate(parts))
        return np.asarray(rows,dtype=np.float32)
