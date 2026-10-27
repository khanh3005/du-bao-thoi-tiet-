"""App chỉ gọi inference chung, không tự viết lại preprocessing."""
import pandas as pd
import streamlit as st
import sys
from pathlib import Path
from config import CLASSES, MODEL_PATH, MODELS, REPORTS
from utils import load_bundle, file_hash, predict_image_bytes

st.set_page_config(page_title='Phân loại ảnh thời tiết', page_icon='🌦️')
st.title('Phân loại ảnh thời tiết')

@st.cache_resource
def cached_bundle(model_path, model_digest):
    # Digest là cache key: thay model sẽ tải lại dù tên file không đổi.
    return load_bundle(Path(model_path))

try:
    experimental = '--experimental' in sys.argv
    selected_path = MODELS / 'experimental_feature_model.pkl' if experimental else MODEL_PATH
    digest = file_hash(selected_path)
    bundle = cached_bundle(str(selected_path), digest)
except Exception as e:
    st.error(f'Không tải được model: {e}')
    st.info('Chạy check_setup.py → preprocessing.py → train_models.py theo README.')
    st.stop()

st.caption(f"Model: {bundle['name']} | Train: {bundle['created_at']} | ID: {digest[:12]}")
if experimental:
    st.warning('Bản nghiên cứu đặc trưng màu/đường nét: mới đánh giá bằng Train CV, chưa có Test độc lập mới.')
st.caption('5 lớp: cloudy, foggy, rainy, shine, sunrise. Ảnh khác các lớp này vẫn có thể bị gán một nhãn.')
upload = st.file_uploader('Chọn ảnh thời tiết', type=['jpg','jpeg','png','webp','bmp'])
if upload is not None:
    data = upload.getvalue()
    try:
        st.image(data, caption=upload.name, use_container_width=True)
    except Exception:
        st.error('File tải lên không phải ảnh đọc được. Hãy kiểm tra lại file.')
        st.stop()
    if st.button('Dự đoán', type='primary'):
        try:
            pred, confidence, probabilities = predict_image_bytes(bundle,data)
            st.subheader(f'Kết quả: {CLASSES[pred[0]]}')
            if confidence is not None:
                st.metric('Xác suất model gán cho nhãn dự đoán', f'{100*confidence[0]:.1f}%')
                st.bar_chart(pd.DataFrame({'class':CLASSES,'probability':probabilities[0]}).set_index('class'))
                st.caption('Đây là xác suất ước lượng của model, không phải bảo đảm dự đoán đúng.')
            else:
                st.caption('Model không cung cấp xác suất; không chuyển điểm số thành phần trăm.')
        except Exception as e:
            st.error(str(e))

eval_dir = REPORTS / 'all_models_evaluation'
summary_file = eval_dir / 'all_models_test_summary.csv'
grid_file = eval_dir / 'combined_8_confusion_matrices.png'
if summary_file.exists() and grid_file.exists():
    with st.expander('📊 Bảng so sánh và Ma trận nhầm lẫn của 8 mô hình'):
        st.write('### Bảng tổng hợp hiệu năng 8 mô hình trên tập Test')
        st.dataframe(pd.read_csv(summary_file).style.format({
            'test_accuracy': '{:.2%}',
            'test_macro_f1': '{:.2%}',
            'test_weighted_f1': '{:.2%}',
            'cloudy_f1': '{:.2%}',
            'foggy_f1': '{:.2%}',
            'rainy_f1': '{:.2%}',
            'shine_f1': '{:.2%}',
            'sunrise_f1': '{:.2%}'
        }))
        st.write('### Ma trận nhầm lẫn (Confusion Matrix) của 8 mô hình')
        st.image(str(grid_file), use_container_width=True)
