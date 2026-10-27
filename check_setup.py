"""Chạy đầu tiên: môi trường, thư viện và folder."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import importlib
from pathlib import Path
from config import ROOT, DATASET, EXTERNAL, CLASSES


def main():
    errors = []
    print('Python:', sys.version)
    print('Interpreter:', sys.executable)
    print('Project:', ROOT)
    if sys.version_info[:2] != (3, 12):
        errors.append('Dùng Python 3.12 để khớp môi trường đã kiểm tra.')
    for name in ['numpy','pandas','cv2','sklearn','scipy','matplotlib','joblib','xgboost','lightgbm','streamlit']:
        try:
            m = importlib.import_module(name)
            print(name, getattr(m, '__version__', 'OK'))
        except Exception as e:
            errors.append(f'{name}: {e}')
    folders = sorted(p.name for p in DATASET.iterdir() if p.is_dir()) if DATASET.exists() else []
    if folders != list(CLASSES):
        errors.append(f'Dataset phải có đúng 5 folder {CLASSES}; hiện có {folders}')
    for name in CLASSES:
        count = sum(p.is_file() for p in (DATASET / name).rglob('*'))
        print(f'{name}: {count} files')
        if count < 10:
            errors.append(f'{name}: quá ít ảnh hoặc thiếu folder.')
    if not EXTERNAL.exists():
        errors.append('Thiếu alien_test/.')
    if errors:
        print('\nKHÔNG ĐẠT:\n' + '\n'.join(errors))
        return 1
    print('\nCHECK SETUP: PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
