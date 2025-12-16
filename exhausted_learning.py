import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random
import os
from sklearn.preprocessing import LabelEncoder
import itertools
import time
import warnings

warnings.filterwarnings('ignore')

# 设置工作目录
base_dir = ''
os.chdir(base_dir)

# 新增：设置训练结果保存路径
training_dir = os.path.join(base_dir, 'training')
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

# 读取训练集、内部验证集、外部验证集的正确标签
datasets = {
    'train_set_': 'train_set.csv',
    'inter_validation_': 'Inter_validation.csv',
    'out_validation_': 'Out_validation.csv'
}

# 初始化 LabelEncoder
le = LabelEncoder()
all_y_values = []  # 用于收集所有数据集的标签

# 读取并编码标签
for key in datasets.values():
    data = pd.read_csv(key)
    y_values = data['group'].values
    all_y_values.extend(y_values)

le.fit(all_y_values)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 映射标签
label_map = {1: 0, 2: 1, 4: 2}

# 优化超参数组合（基于数据规模调整）
hidden_size_options = [
    [9, 6],
    [9, 6, 3],
    [12, 8, 4],
    [16, 12, 8],
    [9, 7, 5, 3]
]

learning_rate_options = [0.01, 0.005, 0.001, 0.0005]
batch_size_options = [16, 32]
dropout_rate_options = [0.1, 0.2, 0.3]
num_epochs_options = [300, 400]
weight_decay_options = [1e-4, 1e-3]

# 生成所有可能的参数组合
param_combinations = []
for hidden_sizes, lr, batch_size, dropout, epochs, wd in itertools.product(
        hidden_size_options, learning_rate_options, batch_size_options,
        dropout_rate_options, num_epochs_options, weight_decay_options
):
    param_combinations.append({
        'hidden_sizes': hidden_sizes,
        'num_classes': 3,
        'batch_size': batch_size,
        'learning_rate': lr,
        'num_epochs': epochs,
        'dropout_rate': dropout,
        'weight_decay': wd
    })

print(f"Generated {len(param_combinations)} parameter combinations")


# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# 数据集类
class MyDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids[idx]


# 神经网络模型
class ImprovedNeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int, dropout_rate: float = 0.3):
        super(ImprovedNeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.SiLU())
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout(dropout_rate))

        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 评估模型
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_true_labels, all_predictions, all_probabilities


# 计算指标
def calculate_metrics(y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    sensitivity = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # 计算 specificity（特异性）和 NPV（阴性预测值）
    cm = confusion_matrix(y_true, y_pred)
    specificity = np.zeros(cm.shape[0])
    npv = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (FP + FN + TP)
        specificity[i] = TN / (TN + FP) if (TN + FP) > 0 else 0
        npv[i] = TN / (TN + FN) if (TN + FN) > 0 else 0
    specificity = np.mean(specificity)
    npv = np.mean(npv)

    return accuracy, auc, sensitivity, specificity, precision, npv, f1


# 计算置信区间
def calculate_confidence_intervals(y_true, y_pred, y_proba, num_bootstrap=1000, alpha=0.95):
    metrics = np.zeros((num_bootstrap, 7))
    n = len(y_true)

    for i in range(num_bootstrap):
        indices = resample(range(n), n_samples=n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        y_proba_boot = np.array(y_proba)[indices]

        metrics[i] = calculate_metrics(y_true_boot, y_pred_boot, y_proba_boot)

    lower_bounds = np.percentile(metrics, (1 - alpha) / 2 * 100, axis=0)
    upper_bounds = np.percentile(metrics, (1 + alpha) / 2 * 100, axis=0)
    return lower_bounds, upper_bounds



def train_single_fold(train_loader, val_loader, params, input_size, patience=20):
    """训练单个折的模型"""
    model = ImprovedNeuralNet(input_size, params['hidden_sizes'],
                              params['num_classes'], params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'],
                           weight_decay=params['weight_decay'])

    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(params['num_epochs']):
        # 训练
        model.train()
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        # 早停
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_model_state, best_val_accuracy


# 在训练集上进行交叉验证评估参数
def evaluate_param_with_cv(X_train, y_train, params, n_folds=5, random_state=42):
    """使用训练集进行交叉验证评估参数组合"""
    set_random_seed(random_state)

    # 准备训练集数据
    train_dataset = MyDataset(X_train, y_train, np.arange(len(y_train)))

    # 使用分层K折交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # 创建数据加载器
        train_subsampler = Subset(train_dataset, train_idx)
        val_subsampler = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subsampler, batch_size=params['batch_size'],
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subsampler, batch_size=params['batch_size'],
                                shuffle=False, num_workers=0)

        # 训练单折
        _, fold_accuracy = train_single_fold(train_loader, val_loader, params, X_train.shape[1])
        fold_accuracies.append(fold_accuracy)

        # 释放内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 计算平均和标准差
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    return mean_accuracy, std_accuracy, fold_accuracies


# 使用最佳参数训练最终模型
def train_final_model(X_train, y_train, X_val, y_val, best_params, ids_train, ids_val):
    """使用最佳参数训练最终模型"""
    input_size = X_train.shape[1]

    # 创建数据加载器
    train_dataset = MyDataset(X_train, y_train, ids_train)
    val_dataset = MyDataset(X_val, y_val, ids_val)

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'],
                            shuffle=False, num_workers=0)

    # 训练最终模型
    best_model_state, best_val_accuracy = train_single_fold(
        train_loader, val_loader, best_params, input_size, patience=20
    )

    # 创建并加载最佳模型
    final_model = ImprovedNeuralNet(input_size, best_params['hidden_sizes'],
                                    best_params['num_classes'], best_params['dropout_rate']).to(device)
    final_model.load_state_dict(best_model_state)

    return final_model, best_val_accuracy


# 评估和保存结果的函数
def evaluate_and_save_results(model, loader, filename_prefix, ids, true_labels, combo_name, params, seed):
    acc, true_labels, preds, probs = evaluate_model(model, loader)

    # 将准确率转换为小数形式
    acc_decimal = acc / 100

    # 计算其他指标
    metrics = calculate_metrics(true_labels, preds, probs)
    lower_bounds, upper_bounds = calculate_confidence_intervals(true_labels, preds, probs)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1 Score'],
        'Value': [acc_decimal] + list(metrics[1:]),
        'Lower Bound': lower_bounds,
        'Upper Bound': upper_bounds
    })

    results_df = pd.DataFrame({
        'ID': ids,
        'True Label': true_labels,
        'Prediction': preds,
        'Probability': [list(proba) for proba in probs]
    })

    # 简化文件名
    param_str = f"hidden{params['hidden_sizes']}_lr{params['learning_rate']}_bs{params['batch_size']}_do{params['dropout_rate']}_wd{params['weight_decay']}"

    # 保存结果
    results_df.to_csv(os.path.join(training_dir,
                                   f'{combo_name}_{filename_prefix}_results_seed_{seed}_{param_str}.csv'),
                      index=False)

    cm = confusion_matrix(true_labels, preds)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(training_dir,
                              f'{combo_name}_{filename_prefix}_confusion_matrix_seed_{seed}_{param_str}.csv'),
                 index=False)

    return {
        'Combo': combo_name,
        'Params': str(params),
        'Set': filename_prefix,
        'Accuracy': acc_decimal,
        'AUC': metrics[1],
        'Sensitivity': metrics[2],
        'Specificity': metrics[3],
        'Precision': metrics[4],
        'NPV': metrics[5],
        'F1 Score': metrics[6]
    }


# 主函数
def main():
    # 确定一个随机种子
    SEED = 42
    set_random_seed(SEED)

    # 设置交叉验证折数
    N_FOLDS = 5

    all_results = []
    param_performance = []  # 记录每个参数组合的性能
    cv_results = []  # 记录交叉验证结果

    # 获取文件夹下所有数据文件并按文件名分组
    files = os.listdir(base_dir)
    file_groups = {}

    for f in files:
        if any(f.startswith(prefix) for prefix in datasets.keys()):
            key = '_'.join(f.split('_')[2:])
            file_groups.setdefault(key, []).append(f)

    # 确保每组都有完整的对应文件
    valid_groups = {k: v for k, v in file_groups.items() if len(v) == 3}

    print(f"Found {len(valid_groups)} valid combinations to process")
    print(f"Total parameter combinations to try: {len(param_combinations)}")
    print(f"Using {N_FOLDS}-fold cross validation on TRAINING SET only")
    print(f"Training set size: ~242 samples, Internal validation: 231 samples, External validation: 132 samples")

    # 训练所有有效的组合
    for combo_idx, (combo_name, files) in enumerate(valid_groups.items(), 1):
        print(f"\n{'=' * 60}")
        print(f"[{combo_idx}/{len(valid_groups)}] Processing combination: {combo_name}")
        print(f"{'=' * 60}")

        # 读取数据
        X_train, Y_train, ids_train = None, None, None
        X_inter_validation, Y_inter_validation, ids_inter_validation = None, None, None
        X_out_validation, Y_out_validation, ids_out_validation = None, None, None

        for f in files:
            if 'train_set' in f:
                train_data = pd.read_csv(f)
                train_set = pd.read_csv(os.path.join(base_dir, "train_set.csv"))
                X_train, Y_train = train_data.values, train_set['group'].values
                ids_train = train_set['ID'].values
                print(f"  Training set loaded: {X_train.shape[0]} samples")
            elif 'inter_validation' in f:
                inter_validation_data = pd.read_csv(f)
                inter_set = pd.read_csv(os.path.join(base_dir, "Inter_validation_re.csv"))
                X_inter_validation, Y_inter_validation = inter_validation_data.values, inter_set['group'].values
                ids_inter_validation = inter_set['ID'].values
                print(f"  Internal validation set loaded: {X_inter_validation.shape[0]} samples")
            elif 'out_validation' in f:
                out_validation_data = pd.read_csv(f)
                out_set = pd.read_csv(os.path.join(base_dir, "Out_validation_re.csv"))
                X_out_validation, Y_out_validation = out_validation_data.values, out_set['group'].values
                ids_out_validation = out_set['ID'].values
                print(f"  External validation set loaded: {X_out_validation.shape[0]} samples")

        # 映射标签
        Y_train = np.vectorize(label_map.get)(Y_train)
        Y_inter_validation = np.vectorize(label_map.get)(Y_inter_validation)
        Y_out_validation = np.vectorize(label_map.get)(Y_out_validation)

        # 为每个组合初始化最佳模型和准确率
        combo_best_cv_accuracy = 0
        combo_best_params = None
        combo_best_cv_scores = None

        # 记录开始时间
        combo_start_time = time.time()


        if len(param_combinations) > 50:
            print(f"  Randomly sampling 50 parameter combinations for efficiency")
            sampled_params = random.sample(param_combinations, 50)
        else:
            sampled_params = param_combinations

        print(f"  Evaluating {len(sampled_params)} parameter combinations using {N_FOLDS}-fold CV on training set")

        # 遍历参数组合，在训练集上进行交叉验证评估
        for param_idx, params in enumerate(sampled_params, 1):
            # 每5个参数组合打印一次进度
            if param_idx % 5 == 0 or param_idx == 1:
                elapsed_time = time.time() - combo_start_time
                eta = (elapsed_time / param_idx) * (len(sampled_params) - param_idx)
                print(f"  [{param_idx}/{len(sampled_params)}] Elapsed: {elapsed_time / 60:.1f}m, ETA: {eta / 60:.1f}m")

            # 在训练集上进行交叉验证评估参数
            cv_mean_acc, cv_std_acc, fold_accuracies = evaluate_param_with_cv(
                X_train, Y_train, params, n_folds=N_FOLDS, random_state=SEED
            )

            # 保存交叉验证结果
            cv_results.append({
                'combo_name': combo_name,
                'params': str(params),
                'cv_mean_accuracy': cv_mean_acc,
                'cv_std_accuracy': cv_std_acc,
                'fold_accuracies': fold_accuracies,
                'params_dict': params
            })

            # 记录参数性能
            param_performance.append({
                'combo_name': combo_name,
                'params': params,
                'cv_mean_accuracy': cv_mean_acc,
                'cv_std_accuracy': cv_std_acc
            })

            # 更新最佳参数
            if cv_mean_acc > combo_best_cv_accuracy:
                combo_best_cv_accuracy = cv_mean_acc
                combo_best_params = params
                combo_best_cv_scores = fold_accuracies
                print(f"    New best CV accuracy: {cv_mean_acc:.2f}% ± {cv_std_acc:.2f}%")

        # 使用最佳参数训练最终模型
        if combo_best_params:
            print(f"\n  Best parameters for {combo_name}:")
            print(f"    Hidden sizes: {combo_best_params['hidden_sizes']}")
            print(f"    Learning rate: {combo_best_params['learning_rate']}")
            print(f"    Batch size: {combo_best_params['batch_size']}")
            print(f"    Dropout: {combo_best_params['dropout_rate']}")
            print(f"    Training CV Accuracy: {combo_best_cv_accuracy:.2f}% ± {np.std(combo_best_cv_scores):.2f}%")
            print(f"    Fold scores: {[f'{score:.1f}%' for score in combo_best_cv_scores]}")

            # 使用最佳参数在训练集上训练最终模型，用内部验证集进行早停
            final_model, final_val_acc = train_final_model(
                X_train, Y_train,
                X_inter_validation, Y_inter_validation,
                combo_best_params,
                ids_train, ids_inter_validation
            )

            print(f"    Internal validation accuracy: {final_val_acc:.2f}%")

            # 在训练集、内部验证集和外部验证集上评估
            train_dataset = MyDataset(X_train, Y_train, ids_train)
            inter_dataset = MyDataset(X_inter_validation, Y_inter_validation, ids_inter_validation)
            out_dataset = MyDataset(X_out_validation, Y_out_validation, ids_out_validation)

            train_loader = DataLoader(train_dataset, batch_size=combo_best_params['batch_size'],
                                      shuffle=False, num_workers=0)
            inter_loader = DataLoader(inter_dataset, batch_size=combo_best_params['batch_size'],
                                      shuffle=False, num_workers=0)
            out_loader = DataLoader(out_dataset, batch_size=combo_best_params['batch_size'],
                                    shuffle=False, num_workers=0)

            # 评估和保存结果
            train_results = evaluate_and_save_results(final_model, train_loader, 'Train_Set',
                                                      ids_train, Y_train, combo_name,
                                                      combo_best_params, SEED)
            inter_results = evaluate_and_save_results(final_model, inter_loader, 'Inter_Validation',
                                                      ids_inter_validation, Y_inter_validation,
                                                      combo_name, combo_best_params, SEED)
            out_results = evaluate_and_save_results(final_model, out_loader, 'Out_Validation',
                                                    ids_out_validation, Y_out_validation,
                                                    combo_name, combo_best_params, SEED)

            all_results.extend([train_results, inter_results, out_results])

            # 保存模型
            model_save_path = os.path.join(training_dir, f'{combo_name}_best_model_cv.pth')
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'params': combo_best_params,
                'cv_scores': combo_best_cv_scores,
                'cv_mean_accuracy': combo_best_cv_accuracy,
                'internal_val_accuracy': final_val_acc
            }, model_save_path)
            print(f"    Model saved to {model_save_path}")

    # 保存所有结果
    # 保存交叉验证结果
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(training_dir, 'cross_validation_results_train_only.csv'), index=False)

    # 保存参数性能
    performance_df = pd.DataFrame(param_performance)
    performance_df.to_csv(os.path.join(training_dir, 'parameter_performance_cv_train_only.csv'), index=False)

    # 汇总所有组合的结果
    results_summary = pd.DataFrame(all_results)
    results_summary.to_csv(os.path.join(training_dir, 'all_combinations_metrics_summary_cv_train_only.csv'),
                           index=False)

    # 创建性能对比表格
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS (Training Set CV Approach)")
    print("=" * 80)

    # 输出每个组合的性能对比
    for i in range(0, len(all_results), 3):
        if i + 2 < len(all_results):
            combo = all_results[i]['Combo']
            train_acc = all_results[i]['Accuracy']
            inter_acc = all_results[i + 1]['Accuracy']
            out_acc = all_results[i + 2]['Accuracy']

            print(f"\nCombo: {combo}")
            print(f"  Training Set Accuracy:    {train_acc:.3f}")
            print(f"  Internal Validation Acc:  {inter_acc:.3f}")
            print(f"  External Validation Acc:  {out_acc:.3f}")
            print(f"  Generalization Gap (Internal - External): {inter_acc - out_acc:.3f}")

    print("\n" + "=" * 80)
    print("Training completed with training set cross validation!")
    print(f"All results and metrics saved to {training_dir}")
    print(f"Processed {len(valid_groups)} combinations")
    print("=" * 80)


if __name__ == "__main__":
    main()