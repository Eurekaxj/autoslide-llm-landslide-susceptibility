import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# sklearn imports
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# xgboost and joblib
from xgboost import XGBClassifier
from joblib import dump, load

# ========== 全局变量（用于存储过程中的中间结果）==========
result_dir = r"D:\XianData\Result\XGBoost Result"
predictres_dir = r"D:\XianData\Verify\XGBoost"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(predictres_dir, exist_ok=True)


excel_writer = pd.ExcelWriter(os.path.join(result_dir, f"metrics.xlsx"), engine='openpyxl')

# 特征列 & 标签列
features = ['DEM', 'slope', 'aspect', 'plan', 'profile',
            'rainfall', 'NDVI', 'land_use', 'road', 'river', 'temyear','roughness']
label_col = 'landslide'

# 参数网格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 测试集路径
prediction_file = r"D:\XianData\Verify\VerifyData.csv"

# 存储结果的全局变量
metrics_list = []
tprs = []
aucs = []
best_params_list = []
model_paths = []
mean_fpr = np.linspace(0, 1, 100)


def train_model(round_num):
    """
    训练单轮模型并保存。
    """
    print(f"===== 第 {round_num} 轮训练 =====")

    file_path = fr"D:\XianData\Data\DataSet__{round_num}.xlsx"
    df = pd.read_excel(file_path)

    # 平衡样本
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0].sample(n=len(df_pos), random_state=42)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 数据准备
    X = df_balanced[features]
    y = df_balanced[label_col]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # 网格搜索调参
    print("正在进行网格搜索调参...")
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                               cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    print(f"第 {round_num} 轮最佳参数: {best_params}")

    model = grid_search.best_estimator_

    # 保存模型
    model_path = os.path.join(result_dir, f"XGBModel_Round{round_num}.joblib")
    dump(model, model_path)
    model_paths.append({'round': round_num, 'path': model_path, 'auc': None})

    return model, X_train, X_test, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test, round_num):
    """
    评估模型性能并绘图。
    """

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_cm = confusion_matrix(y_test, y_test_pred)

    metrics_list.append({
        'Round': round_num,
        'Test_Accuracy': test_acc,
        'Test_Precision': test_prec,
        'Test_Recall': test_rec,
        'Test_F1-Score': test_f1,
        'Test_AUC': test_auc
    })

    # 更新 AUC 到 model_paths
    for m in model_paths:
        if m['round'] == round_num:
            m['auc'] = test_auc

    # 绘制 ROC 曲线
    test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(test_fpr, test_tpr, label=f'Testing ROC Curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Round {round_num}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f"ROC_Round_{round_num}.png"))
    plt.close()

    # 混淆矩阵

    plt.subplot(1, 2, 2)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Testing Confusion Matrix - Round {round_num}")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"CM_Round_{round_num}.png"))
    plt.close()

    # 存储本轮的 tpr 和 auc
    interp_tpr = np.interp(mean_fpr, test_fpr, test_tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(test_auc)


def predict(model, scaler, round_num):
    """
    使用当前模型预测验证集并保存结果。
    """
    predict_df = pd.read_csv(prediction_file)
    X_predict = predict_df[features]
    X_predict_scaled = scaler.transform(X_predict)

    predict_proba = model.predict_proba(X_predict_scaled)[:, 1]

    predict_df['Predicted_Probability'] = predict_proba

    verify_output_path = os.path.join(predictres_dir, f"prediction_round{round_num}.csv")
    predict_df.to_csv(verify_output_path, index=False, encoding='utf-8-sig')

    print(f"✅ 验证集预测完成，结果已保存为：{verify_output_path}")


def main():
    num_rounds = 100  # 设置训练轮数

    for round_num in range(1, num_rounds + 1):
        model, X_train, X_test, y_train, y_test, scaler = train_model(round_num)
        evaluate_model(model, X_train, X_test, y_train, y_test, round_num)
        predict(model, scaler, round_num)

    # 保存指标到 Excel
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.loc['Average'] = df_metrics.mean(numeric_only=True)
    df_metrics.to_excel(excel_writer, sheet_name='Metrics', index=False)

    # 绘制平均 ROC 曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = {:.3f} ± {:.3f})'.format(mean_auc, std_auc),
             lw=2, alpha=0.8)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label='±1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve Across Rounds')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, f"Mean_ROC.png"))
    plt.close()

    excel_writer.close()
    print("✅ 所有XGBoost模型训练完成，指标与图表已导出。")


if __name__ == "__main__":
    main()