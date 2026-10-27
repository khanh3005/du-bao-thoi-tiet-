"""Kiểm thử Streamlit bằng AppTest; upload được giả lập, model chạy thật.
Không dùng file này để đánh giá/chọn model. Chạy sau train_models.py.
"""
import io
import json
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from config import ROOT, EXTERNAL, REPORTS, CLASSES
from utils import load_bundle, predict_image_bytes


def main():
    sys.path.insert(0,str(ROOT))
    bundle = load_bundle()
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    results=[]
    # Từng ảnh external, kể cả ảnh model đoán sai: app phải khớp script.
    for path in sorted(EXTERNAL.iterdir()):
        if not path.is_file():
            continue
        data=path.read_bytes()
        upload=io.BytesIO(data)
        upload.name=path.name
        pred,confidence,_=predict_image_bytes(bundle,data)
        with patch('streamlit.file_uploader', return_value=upload):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
            assert not at.exception, at.exception
            at.button[0].click().run()
            assert not at.exception, at.exception
            assert not at.error, at.error
            assert at.subheader[0].value == f'Kết quả: {CLASSES[pred[0]]}'
            if confidence is not None:
                assert at.metric[0].value == f'{100*confidence[0]:.1f}%'
        results.append({'image':path.name,'script_predicted':CLASSES[pred[0]], 'app_equals_script':True})
    upload=io.BytesIO(b'broken image')
    upload.name='broken.jpg'
    with patch('streamlit.file_uploader', return_value=upload):
        bad=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        assert not bad.exception
        assert len(bad.error)==1
    pd.DataFrame(results).to_csv(REPORTS/'app_script_parity.csv',index=False)
    output={'method':'Streamlit AppTest with mocked uploader, real button/prediction rendering',
            'images_checked':len(results), 'matches':len(results),
            'initial_app_exceptions':0,'invalid_upload_handled':True,
            'note':'Not a manual browser or Windows test.'}
    (REPORTS/'app_test_result.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(output)

if __name__ == '__main__':
    main()
