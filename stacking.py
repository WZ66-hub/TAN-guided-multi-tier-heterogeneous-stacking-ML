import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# 设置当前目录
os.chdir('')

# 定义读取文件的函数
def load_probabilities(file_paths, columns_range_list):
    probabilities = []
    for file, col_range in zip(file_paths, columns_range_list):
        data = pd.read_csv(file)
        probabilities.append(data.iloc[:, col_range].values)
    return probabilities

# 定义读取数据集和标签的函数
def load_data_and_labels(data_file, labels_column):
    data = pd.read_csv(data_file)
    return data, data[labels_column].values

# 定义映射标签的函数
def map_labels(labels, label_map):
    return np.vectorize(label_map.get)(labels)

# 文件路径和列范围定义
prob_files_train = [
    'MLP_Train_best_predictions.csv', 'RF_train_predictions.csv', 'XGB_prediction_train.csv',
    'Bagging_train_predictions.csv', 'DT_Train_predictions_with_proba.csv',
    'LightGBM_Training_Predictions.csv', 'Logistic_train_set_predictions.csv',
    'SVM_train_predictions.csv'
]

columns_range_train = [
    range(3, 6), range(3, 6), range(3, 6), range(3, 6), range(14, 17), range(2, 5), range(3, 6), range(3, 6)
]

prob_files_inter_validation = [
    'MLP_Inter Validation_best_predictions.csv', 'RF_inter_validation_predictions.csv',
    'XGB_prediction_inter_validation.csv', 'Bagging_inter_validation_predictions.csv',
    'DT_Inter Validation_predictions_with_proba.csv', 'LightGBM_Inter Validation_Predictions.csv',
    'Logistic_inter_validation_set_predictions.csv', 'SVM_inter_validation_predictions.csv'
]

columns_range_inter_validation = [
    range(3, 6), range(3, 6), range(3, 6), range(3, 6), range(14, 17), range(2, 5), range(3, 6), range(3, 6)
]

prob_files_out_validation = [
    'MLP_Out Validation_best_predictions.csv', 'RF_out_validation_predictions.csv',
    'XGB_prediction_out_validation.csv', 'Bagging_out_validation_predictions.csv',
    'DT_Out Validation_predictions_with_proba.csv', 'LightGBM_Out Validation_Predictions.csv',
    'Logistic_out_validation_set_predictions.csv', 'SVM_out_validation_predictions.csv'
]

columns_range_out_validation = [
    range(3, 6), range(3, 6), range(3, 6), range(3, 6), range(14, 17), range(2, 5), range(3, 6), range(3, 6)
]



# 读取训练集和内部验证集的预测概率
mlp_prob_train, rf_prob_train, xgb_prob_train, bagging_prob_train, dt_prob_train, lightGBM_prob_train, logistic_prob_train, svm_prob_train = load_probabilities(prob_files_train, columns_range_train)
mlp_prob_inter_validation, rf_prob_inter_validation, xgb_prob_inter_validation, bagging_prob_inter_validation, dt_prob_inter_validation, lightGBM_prob_inter_validation, logistic_prob_inter_validation, svm_prob_inter_validation = load_probabilities(prob_files_inter_validation, columns_range_inter_validation)
mlp_prob_out_validation, rf_prob_out_validation, xgb_prob_out_validation, bagging_prob_out_validation, dt_prob_out_validation, lightGBM_prob_out_validation, logistic_prob_out_validation, svm_prob_out_validation = load_probabilities(prob_files_out_validation, columns_range_out_validation)


# 读取数据集和标签
train_data, y_train = load_data_and_labels('train_set.csv', 'group')
inter_validation_data, y_inter_validation = load_data_and_labels('Inter_validation.csv', 'group')
out_validation_data, y_out_validation = load_data_and_labels('Out_validation.csv', 'group')

# 标签映射
label_map = {1: 0, 2: 1, 4: 2}
y_train = map_labels(y_train, label_map)
y_inter_validation = map_labels(y_inter_validation, label_map)
y_out_validation = map_labels(y_out_validation, label_map)


# 模型组合
model_names = ['MLP', 'RF', 'XGB', 'Bagging', 'DT', 'LightGBM', 'Logistic', 'SVM']
all_combinations = list(itertools.combinations(range(len(model_names)), 3))

# 获取概率数据
prob_train = [mlp_prob_train, rf_prob_train, xgb_prob_train, bagging_prob_train, dt_prob_train, lightGBM_prob_train, logistic_prob_train, svm_prob_train]
prob_inter_validation = [mlp_prob_inter_validation, rf_prob_inter_validation, xgb_prob_inter_validation, bagging_prob_inter_validation, dt_prob_inter_validation, lightGBM_prob_inter_validation, logistic_prob_inter_validation, svm_prob_inter_validation]
prob_out_validation = [mlp_prob_out_validation, rf_prob_out_validation, xgb_prob_out_validation, bagging_prob_out_validation, dt_prob_out_validation, lightGBM_prob_out_validation, logistic_prob_out_validation, svm_prob_out_validation]

# 定义保存组合概率数据的函数
def save_combination_probabilities(prob_data, combinations, prefix, output_dir, suffix='_comb.csv'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for combo in combinations:
        combined_probs = np.hstack([prob_data[i] for i in combo])
        file_name = f"{prefix}_{'_'.join(model_names[i] for i in combo)}{suffix}"
        output_path = os.path.join(output_dir, file_name)
        pd.DataFrame(combined_probs).to_csv(output_path, index=False)

# 指定保存文件的目录
output_directory =''

# 保存训练集组合概率数据
save_combination_probabilities(prob_train, all_combinations, 'train_set', output_directory)

# 保存内部验证集组合概率数据
save_combination_probabilities(prob_inter_validation, all_combinations, 'inter_validation', output_directory)

# 保存外部验证集组合概率数据
save_combination_probabilities(prob_out_validation, all_combinations, 'out_validation', output_directory)

