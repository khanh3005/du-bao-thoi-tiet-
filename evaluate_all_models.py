"""Huấn luyện và xuất đầy đủ Classification Report & Confusion Matrix cho toàn bộ 8 mô hình trên tập Test."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from config import ROOT, PROCESSED, REPORTS, CLASSES, N_JOBS, SEED
from train_models import candidates


def slugify(name):
    return name.lower().replace(' ', '_')


def main():
    out_dir = REPORTS / 'all_models_evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BẮT ĐẦU ĐÁNH GIÁ VÀ XUẤT BÁO CÁO CHO ĐỦ 8 MÔ HÌNH HỌC MÁY")
    print("=" * 70)

    # 1. Nạp dữ liệu đã xử lý
    X_train = np.load(PROCESSED / 'X_train.npy', allow_pickle=False)
    y_train = np.load(PROCESSED / 'y_train.npy', allow_pickle=False)
    X_test = np.load(PROCESSED / 'X_test.npy', allow_pickle=False)
    y_test = np.load(PROCESSED / 'y_test.npy', allow_pickle=False)

    # 2. Đọc cấu hình model tốt nhất từ model_comparison.csv
    comp_path = REPORTS / 'model_comparison.csv'
    if not comp_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {comp_path}. Vui lòng chạy train_models.py trước.")
    
    comp_df = pd.read_csv(comp_path)
    winner_indices = dict(zip(comp_df['model'], comp_df['candidate']))

    registry = candidates()
    summary_rows = []
    cm_dict = {}

    labels = list(range(len(CLASSES)))

    print(f"{'Mô hình':<25} | {'Test Acc':<10} | {'Test Macro F1':<14} | {'Thời gian fit'}")
    print("-" * 70)

    with threadpool_limits(limits=N_JOBS):
        for name in comp_df['model']:
            cand_idx = int(winner_indices[name])
            model = registry[name][cand_idx]
            slug = slugify(name)

            # Fit trên toàn bộ tập train
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start_time

            # Dự đoán trên tập test
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, labels=labels, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_test, preds, labels=labels, average='weighted', zero_division=0)

            # Classification report
            report_dict = classification_report(
                y_test, preds, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0
            )
            report_text = classification_report(
                y_test, preds, labels=labels, target_names=CLASSES, zero_division=0
            )

            # Lưu báo cáo riêng của từng model
            (out_dir / f'{slug}_classification_report.txt').write_text(report_text, encoding='utf-8')
            pd.DataFrame(report_dict).T.to_csv(out_dir / f'{slug}_classification_report.csv')

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds, labels=labels)
            cm_dict[name] = (cm, acc, macro_f1)
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / f'{slug}_confusion_matrix.csv')

            # Vẽ Confusion Matrix đơn lẻ
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap='Blues')
            for i in labels:
                for j in labels:
                    val = cm[i, j]
                    color = 'white' if val > cm.max() / 2 else 'black'
                    ax.text(j, i, str(val), ha='center', va='center', color=color, fontweight='bold')
            ax.set(xticks=labels, yticks=labels, xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel='Dự đoán (Predicted)', ylabel='Thực tế (True)',
                   title=f'{name}\nTest Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%')
            fig.tight_layout()
            fig.savefig(out_dir / f'{slug}_confusion_matrix.png', dpi=150)
            plt.close(fig)

            # Thu thập điểm per-class F1
            per_class_f1 = {f'{c}_f1': report_dict[c]['f1-score'] for c in CLASSES}

            summary_rows.append({
                'model': name,
                'test_accuracy': acc,
                'test_macro_f1': macro_f1,
                'test_weighted_f1': weighted_f1,
                **per_class_f1,
                'fit_time_seconds': round(fit_time, 2)
            })

            print(f"{name:<25} | {acc*100:>8.2f}% | {macro_f1*100:>12.2f}% | {fit_time:>6.2f}s")

    # 3. Lưu bảng tổng hợp
    summary_df = pd.DataFrame(summary_rows).sort_values('test_macro_f1', ascending=False)
    summary_df.to_csv(out_dir / 'all_models_test_summary.csv', index=False)

    # 4. Vẽ ảnh ghép tổng hợp 8 Confusion Matrices (2 hàng x 4 cột)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, (name, (cm, acc, macro_f1)) in enumerate(cm_dict.items()):
        ax = axes[idx]
        ax.imshow(cm, cmap='Blues')
        for i in labels:
            for j in labels:
                val = cm[i, j]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=10, fontweight='bold')
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(CLASSES, rotation=35, ha='right', fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_title(f"{name}\nAcc: {acc*100:.1f}% | F1: {macro_f1*100:.1f}%", fontsize=11, fontweight='bold')
        if idx % 4 == 0:
            ax.set_ylabel('Thực tế (True)', fontsize=10)
        if idx >= 4:
            ax.set_xlabel('Dự đoán (Predicted)', fontsize=10)

    fig.suptitle('SO SÁNH CONFUSION MATRIX CỦA 8 MÔ HÌNH HỌC MÁY TRÊN TẬP TEST', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    combined_img = out_dir / 'combined_8_confusion_matrices.png'
    fig.savefig(combined_img, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH XUẤT BÁO CÁO 8 MÔ HÌNH!")
    print(f"- Thư mục lưu: {out_dir}")
    print(f"- Ảnh ghép 8 ma trận: {combined_img.name}")
    print(f"- Bảng điểm tổng hợp: all_models_test_summary.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()

