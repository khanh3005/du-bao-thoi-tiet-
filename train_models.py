"""Chọn tham số và model CHỈ trên Train. Test mở sau khi khóa lựa chọn."""
import json
import time
import warnings
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import *
from utils import *
from evaluation import save_evaluation


def candidates():
    def scaled(model):
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    # Chốt lưới trước khi mở Test; giữ đủ 8 họ thuật toán.
    return {
        'Logistic Regression': [scaled(LogisticRegression(C=c, max_iter=2500, random_state=SEED)) for c in [0.01, 0.1]],
        'Gaussian Naive Bayes': [GaussianNB(var_smoothing=v) for v in [1e-9, 1e-2]],
        'Random Forest': [RandomForestClassifier(n_estimators=160, max_features='sqrt', min_samples_leaf=m,
                                                n_jobs=N_JOBS, random_state=SEED) for m in [1,3]],
        'SVM Linear': [scaled(SVC(kernel='linear', C=c, probability=True, random_state=SEED, cache_size=512)) for c in [0.001,0.01]],
        'SVM RBF': [scaled(SVC(kernel='rbf', C=c, probability=True, random_state=SEED, cache_size=512)) for c in [1,10]],
        'SVM Polynomial': [scaled(SVC(kernel='poly', C=c, degree=3, coef0=1, probability=True,
                                      random_state=SEED, cache_size=512)) for c in [0.1,1]],
        'XGBoost': [XGBClassifier(n_estimators=120, max_depth=d, learning_rate=0.08, subsample=0.85,
                                  colsample_bytree=0.5, tree_method='hist', max_bin=64,
                                  objective='multi:softprob', eval_metric='mlogloss',
                                  n_jobs=N_JOBS, random_state=SEED) for d in [3,5]],
        'LightGBM': [LGBMClassifier(n_estimators=160, num_leaves=l, learning_rate=0.05,
                                    colsample_bytree=0.7, reg_lambda=1, verbosity=-1,
                                    n_jobs=N_JOBS, random_state=SEED) for l in [7,15]]}


def main():
    make_dirs()
    if (REPORTS / 'final_test_started.json').exists() or (REPORTS / 'final_test_metrics.json').exists():
        raise RuntimeError('Test đã được mở ở run này. Không chạy lại để chọn model theo Test. Xem README.')
    meta = json.loads((PROCESSED / 'metadata.json').read_text())
    if meta['classes'] != list(CLASSES) or meta['preprocess'] != PREPROCESS_SPEC:
        raise ValueError('Metadata không khớp cấu hình.')
    for path, digest in meta['checksums'].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f'Dữ liệu đã thay đổi: {path}. Chạy lại preprocessing trước training.')
    X = np.load(PROCESSED / 'X_train.npy', allow_pickle=False)
    y = np.load(PROCESSED / 'y_train.npy', allow_pickle=False)
    groups = np.load(PROCESSED / 'groups_train.npy', allow_pickle=False)
    cv = list(StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(X,y,groups))
    for a,b in cv:
        assert not (set(groups[a]) & set(groups[b]))
        assert set(y[a]) == set(y[b]) == set(range(5))
    write_json(REPORTS / 'cv_folds.json', [{'train_indices':a.tolist(),'validation_indices':b.tolist()} for a,b in cv])
    registry = candidates()
    spec = {name:[repr(m) for m in models] for name,models in registry.items()}
    write_json(REPORTS / 'search_plan.json', {'dataset_id':meta['dataset_id'], 'metric':'macro_f1',
                                            'cv_folds':CV_FOLDS, 'candidates':spec})
    rows, winners = [], {}
    # Nếu một model lỗi, dừng và sửa; không âm thầm bỏ một trong 8 model.
    with threadpool_limits(limits=N_JOBS):
        for name, models in registry.items():
            for index, model in enumerate(models):
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    scores = cross_validate(model, X, y, cv=cv, scoring={'f1':'f1_macro','acc':'accuracy'},
                                            n_jobs=1, return_train_score=True, error_score='raise')
                convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                if convergence:
                    raise RuntimeError(f'{name} chưa hội tụ: {convergence[0]}')
                row = {'model':name, 'candidate':index, 'cv_macro_f1':float(scores['test_f1'].mean()),
                       'cv_f1_std':float(scores['test_f1'].std()), 'cv_accuracy':float(scores['test_acc'].mean()),
                       'cv_train_accuracy':float(scores['train_acc'].mean()), 'seconds':time.perf_counter()-start,
                       'parameters':repr(model)}
                rows.append(row)
                pd.DataFrame(rows).to_csv(REPORTS / 'cv_all_candidates.csv', index=False)
                print(f"{name} [{index}] CV Macro F1={row['cv_macro_f1']:.4f}, Accuracy={row['cv_accuracy']:.4f}", flush=True)
            best = sorted([r for r in rows if r['model']==name], key=lambda r:(-r['cv_macro_f1'],-r['cv_accuracy'],r['candidate']))[0]
            winners[name] = best
        comparison = pd.DataFrame(winners.values()).sort_values(['cv_macro_f1','cv_accuracy'], ascending=False, kind='stable')
        comparison.to_csv(REPORTS / 'model_comparison.csv', index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ax = comparison.set_index('model')['cv_macro_f1'].plot.barh(figsize=(9,5), xlim=(0,1), xlabel='Train CV Macro F1')
        ax.figure.tight_layout(); ax.figure.savefig(REPORTS/'model_comparison.png', dpi=150); plt.close(ax.figure)
        best = comparison.iloc[0]
        model = clone(registry[best['model']][int(best['candidate'])])
        print('Chốt model từ CV:',best['model'],flush=True)
        model.fit(X,y)
        bundle = {'schema':1, 'model':model, 'name':best['model'], 'classes':list(CLASSES),
                  'preprocess':PREPROCESS_SPEC, 'dataset_id':meta['dataset_id'], 'environment':environment(),
                  'cv_macro_f1':float(best['cv_macro_f1']), 'created_at':datetime.now(timezone.utc).isoformat()}
        temporary = MODELS / 'best_weather_model.tmp'
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(MODEL_PATH)
        metadata = {k:v for k,v in bundle.items() if k!='model'}
        metadata['model_sha256'] = file_hash(MODEL_PATH)
        write_json(MODELS / 'model_metadata.json', metadata)
        # Sau dòng này chỉ báo cáo, không sửa tham số hay chọn lại model.
        write_json(REPORTS / 'final_test_started.json', metadata)
        test_X = np.load(PROCESSED / 'X_test.npy', allow_pickle=False)
        test_y = np.load(PROCESSED / 'y_test.npy', allow_pickle=False)
        predictions, confidence, _ = predict_vectors(bundle, test_X)
        metrics = save_evaluation(test_y, predictions, 'final_test')
        train_accuracy = accuracy_score(y,model.predict(X))
        metrics.update(train_accuracy=train_accuracy, train_test_gap=train_accuracy-metrics['accuracy'],
                       selected_model=best['model'], cv_macro_f1=float(best['cv_macro_f1']), dataset_id=meta['dataset_id'])
        write_json(REPORTS / 'final_test_metrics.json', metrics)
        pd.DataFrame([metrics]).to_csv(REPORTS/'final_test_metrics.csv',index=False)
        frame = pd.read_csv(REPORTS/'test_manifest.csv')
        frame['predicted'] = [CLASSES[i] for i in predictions]
        frame['confidence'] = confidence if confidence is not None else np.nan
        frame['correct'] = predictions == test_y
        frame.to_csv(REPORTS/'test_predictions.csv',index=False)
        print(json.dumps(metrics, indent=2),flush=True)

if __name__ == '__main__':
    main()
