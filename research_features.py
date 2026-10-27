"""Thí nghiệm cải thiện CHỈ Train/CV. Không đọc X_test/y_test, không đánh giá external.
Không thay model app mặc định hoặc báo cáo Final Test của baseline.
"""
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from datetime import datetime,timezone
from config import *
from utils import environment, write_json
from train_models import candidates
from features import WeatherFeatures


def main():
    meta=json.loads((PROCESSED/'metadata.json').read_text())
    X=np.load(PROCESSED/'X_train.npy',allow_pickle=False)
    y=np.load(PROCESSED/'y_train.npy',allow_pickle=False)
    groups=np.load(PROCESSED/'groups_train.npy',allow_pickle=False)
    cv=list(StratifiedGroupKFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED).split(X,y,groups))
    registry=candidates()
    rows=[]
    with threadpool_limits(limits=N_JOBS):
        for name,models in registry.items():
            for index,model in enumerate(models):
                pipeline=Pipeline([('features',WeatherFeatures()),('classifier',clone(model))])
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    s=cross_validate(pipeline,X,y,cv=cv,scoring={'f1':'f1_macro','acc':'accuracy'},
                                     return_train_score=True,error_score='raise',n_jobs=1)
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                    raise RuntimeError(f'{name}: chưa hội tụ.')
                row={'model':name,'candidate':index,'cv_macro_f1':float(s['test_f1'].mean()),
                     'cv_accuracy':float(s['test_acc'].mean()),'cv_f1_std':float(s['test_f1'].std()),
                     'cv_train_accuracy':float(s['train_acc'].mean()),'seconds':time.perf_counter()-start}
                rows.append(row)
                pd.DataFrame(rows).to_csv(REPORTS/'feature_research_all_candidates.csv',index=False)
                print(row,flush=True)
        table=pd.DataFrame(rows).sort_values(['cv_macro_f1','cv_accuracy'],ascending=False,kind='stable')
        table.drop_duplicates('model').to_csv(REPORTS/'feature_research_comparison.csv',index=False)
        best=table.iloc[0]
        chosen=Pipeline([('features',WeatherFeatures()),('classifier',clone(registry[best['model']][int(best['candidate'])]))])
        chosen.fit(X,y)
        bundle={'schema':1,'model':chosen,'name':best['model']+' + HSV/HOG (EXPERIMENTAL)',
                'classes':list(CLASSES),'preprocess':PREPROCESS_SPEC,'dataset_id':meta['dataset_id'],
                'environment':environment(),'cv_macro_f1':float(best['cv_macro_f1']),
                'created_at':datetime.now(timezone.utc).isoformat(),
                'evaluation_status':'Train CV only. No independent final Test. Not the default app model.'}
        joblib.dump(bundle,MODELS/'experimental_feature_model.pkl',compress=3)
        write_json(MODELS/'experimental_feature_metadata.json',{k:v for k,v in bundle.items() if k!='model'})
        print('SAVED EXPERIMENTAL CANDIDATE, NOT FINAL-TESTED:',bundle['name'],flush=True)

if __name__ == '__main__':
    main()
