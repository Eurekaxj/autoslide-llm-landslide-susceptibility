import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# ========== 配置部分 ==========

# 结果输出文件夹
result_dir = r"D:\XianData\Result\Random Forest Result\100+param"
os.makedirs(result_dir, exist_ok=True)

# 时间戳
now_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
excel_writer = pd.ExcelWriter(os.path.join(result_dir, f"{now_str}_metrics.xlsx"), engine='openpyxl')


# 训练轮数
num_rounds = 100

# 特征列
features = ['DEM', 'slope', 'aspect', 'plan', 'profile',
                'rainfall', 'NDVI', 'land_use', 'road', 'river','temyear','roughness']

# 标签列
label_col = 'landslide'

# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [50, 80, 100],  # 减少树的数量上限
    'max_depth': [3, 5, 7],  # 严格限制树深度
    'min_samples_split': [5, 10, 15],  # 增大分裂所需最小样本数
    'min_samples_leaf': [3, 5, 7],  # 增大叶节点最小样本数
    'max_features': [0.3, 0.5, 'sqrt'],  # 限制特征使用比例
    'class_weight': ['balanced', {0: 1, 1: 3}]  # 加强少数类权重
}

# 测试集路径（记得根据自己实际情况修改）
# verify_file = r"D:\GIS_Projects\VerifyData\VerifyData.csv"

# ========== 开始训练循环 ==========

excel_writer = pd.ExcelWriter(os.path.join(result_dir, f"{now_str}_metrics.xlsx"), engine='openpyxl')

metrics_list = []
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
best_params_list = []  # 存储每轮的最佳参数

# # 固定数据集路径
# data_file = r"D:\XianData\Data\DataSet.xls"
# # 读取完整数据集
# df = pd.read_excel(data_file)

for round_num in range(1, num_rounds + 1):
    print(f"===== 第 {round_num} 轮训练 =====")

    # 读取数据
    file_path = fr"D:\XianData\Data\DataSet__{round_num}.xlsx"
    df = pd.read_excel(file_path)

    # 保持滑坡与非滑坡比例1:1
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0].sample(n=len(df_pos), random_state=42)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 分训练/测试集
    X = df_balanced[features]
    y = df_balanced[label_col]

    # 特征归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # # 建模
    # model = RandomForestClassifier(n_estimators=80,  #指定森林中决策树的数量
    #                                max_depth=3,
    #                              random_state=42)
    # model.fit(X_train, y_train)

    # 创建基础模型
    rf = RandomForestClassifier(random_state=42,
                                bootstrap=False,  # 不使用有放回采样
                                oob_score=False)  # 关闭OOB评估
    # 网格搜索
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3,
                               scoring='roc_auc', verbose=1, n_jobs=1)
    grid_search.fit(X_train, y_train)
    # 获取最佳参数
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    print(f"第 {round_num} 轮最佳参数: {best_params}")

    # 使用最佳参数训练模型
    model = grid_search.best_estimator_

    # # 第1轮保存模型
    # if round_num == 1:
    #     model_save_path = os.path.join(result_dir, f"{now_str}_XGBModel_Round1.joblib")
    #     dump(model, model_save_path)

    # 训练集预测
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # 测试集预测
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 训练集评估
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_cm = confusion_matrix(y_train, y_train_pred)

    # 测试集评估
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_cm = confusion_matrix(y_test, y_test_pred)

    metrics_list.append({
        'Round': round_num,
        'Train_Accuracy': train_acc,
        'Train_Precision': train_prec,
        'Train_Recall': train_rec,
        'Train_F1-Score': train_f1,
        'Train_AUC': train_auc,
        'Test_Accuracy': test_acc,
        'Test_Precision': test_prec,
        'Test_Recall': test_rec,
        'Test_F1-Score': test_f1,
        'Test_AUC': test_auc
    })

    # ROC 曲线
    train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
    test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(train_fpr, train_tpr, label=f'Training ROC Curve (AUC = {train_auc:.3f})')
    plt.plot(test_fpr, test_tpr, label=f'Testing ROC Curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Round {round_num}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f"{now_str}_ROC_Round_{round_num}.png"))
    plt.close()

    # 混淆矩阵
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Training Confusion Matrix - Round {round_num}")

    plt.subplot(1, 2, 2)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Testing Confusion Matrix - Round {round_num}")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{now_str}_CM_Round_{round_num}.png"))
    plt.close()

    # 存储本轮的tpr和auc
    interp_tpr = np.interp(mean_fpr, test_fpr, test_tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(test_auc)

# 汇总评估结果
df_metrics = pd.DataFrame(metrics_list)
df_metrics.loc['Average'] = df_metrics.mean(numeric_only=True)
df_metrics.to_excel(excel_writer, sheet_name='Metrics', index=False)

# 平均 ROC 曲线
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
plt.savefig(os.path.join(result_dir, f"{now_str}_Mean_ROC.png"))
plt.close()

excel_writer.close()

print("✅ 所有Random Forest模型训练完成，指标与图表已导出。")

# ========== 用第1轮模型预测测试集 ==========

# print("===== 开始预测验证集 =====")
#
# # 读取测试集
# verify_df = pd.read_csv(verify_file)
#
# # 加载第1轮模型
# model_round1 = load(model_save_path)
#
# # 预测
# X_verify = verify_df[features]
# verify_pred = model_round1.predict(X_verify)
# verify_proba = model_round1.predict_proba(X_verify)[:, 1]
#
# # 加入预测结果
# verify_df['Predicted_Label'] = verify_pred
# verify_df['Predicted_Probability'] = verify_proba
#
# # 保存为xls
# output_verify_path = os.path.join(result_dir, f"{now_str}_Verify_Prediction.csv")
# verify_df.to_csv(output_verify_path, index=False, encoding='utf-8-sig')
#
# print("测试集预测完成，结果已保存为：", output_verify_path)



