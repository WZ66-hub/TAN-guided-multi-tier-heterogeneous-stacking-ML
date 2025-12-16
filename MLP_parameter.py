import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd
import random
import os
import time

os.chdir('')

# 创建结果保存目录
results_dir = 'training_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集类型
DATASET_TYPES = {
    'train_set': 'labeled',
    'inter_validation': 'labeled',
    'out_validation': 'labeled',
    'survival_validation1': 'unlabeled',
    'survival_validation2': 'unlabeled',
    'survival_validation3': 'unlabeled',
    'immune_validation1': 'unlabeled',
    'immune_validation2': 'unlabeled'
}


def calculate_metrics(y_true, y_pred, y_proba):
    """计算有标签数据的评估指标"""
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



class LabeledDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class UnlabeledDataset(Dataset):
    def __init__(self, X, ids=None):
        self.X = torch.FloatTensor(X)
        self.ids = ids if ids is not None else np.arange(len(X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ids[idx]


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



def evaluate_labeled_model(model, data_loader):
    """评估有标签的数据集"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in data_loader:
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



def predict_unlabeled_model(model, data_loader):
    """预测无标签的数据集"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_ids = []

    with torch.no_grad():
        for inputs, ids in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_ids.extend(ids.cpu().numpy())

    return all_ids, all_predictions, all_probabilities


def train_model_with_early_stopping(model, train_loader, val_loader, num_epochs, criterion, optimizer, patience=20):
    """带早停的训练函数"""
    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # 验证阶段
        val_accuracy, _, _, _ = evaluate_labeled_model(model, val_loader)
        val_accuracies.append(val_accuracy)

        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # 早停逻辑
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停在第 {epoch + 1} 轮触发")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_accuracy, train_losses, val_accuracies


# 加载数据
datasets = {
    'train_set': ('', pd.read_excel('')),
    'inter_validation': ('', pd.read_excel("")),
    'out_validation': ('', pd.read_excel('')),
    'survival_validation1': ('', pd.read_excel('')),
    'survival_validation2': ('', pd.read_excel('')),
    'survival_validation3': ('', pd.read_excel('')),
    'immune_validation1': ('', pd.read_excel('')),
    'immune_validation2': ('', pd.read_excel(''))
}


combo_datasets = {}
true_labels = {}
ids = {}

for name, files in datasets.items():
    # 加载特征数据
    combo_datasets[name] = files[0]

    if DATASET_TYPES[name] == 'labeled':

        true_labels[name] = files[1]["group"].values
        ids[name] = files[1]["ID"].values
    else:

        if "ID" in files[1].columns:
            ids[name] = files[1]["ID"].values
        else:

            ids[name] = np.arange(len(files[1]))
        true_labels[name] = None


label_map = {1: 0, 2: 1, 4: 2}
for name in true_labels:
    if true_labels[name] is not None:
        true_labels[name] = np.vectorize(label_map.get)(true_labels[name])


X_train, Y_train = combo_datasets['train_set'].values, true_labels['train_set']
X_inter_validation, Y_inter_validation = combo_datasets['inter_validation'].values, true_labels['inter_validation']
X_out_validation, Y_out_validation = combo_datasets['out_validation'].values, true_labels['out_validation']

# 无标签数据集
X_survival_validation1 = combo_datasets['survival_validation1'].values
X_survival_validation2 = combo_datasets['survival_validation2'].values
X_survival_validation3 = combo_datasets['survival_validation3'].values
X_immune_validation1 = combo_datasets['immune_validation1'].values
X_immune_validation2 = combo_datasets['immune_validation2'].values

# 设置最佳参数
BEST_PARAMS = {
    'hidden_sizes': [9, 6],
    'batch_size': 64,
    'learning_rate': 0.009,
    'num_epochs': 400,
    'dropout_rate': 0.3,
    'weight_decay': 1e-5,
    'patience': 20
}

print("=" * 80)
print("使用已确定的最佳参数训练模型")
print("=" * 80)
print(f"隐藏层结构: {BEST_PARAMS['hidden_sizes']}")
print(f"批大小: {BEST_PARAMS['batch_size']}")
print(f"学习率: {BEST_PARAMS['learning_rate']}")
print(f"Dropout率: {BEST_PARAMS['dropout_rate']}")
print(f"权重衰减: {BEST_PARAMS['weight_decay']}")
print(f"训练轮数: {BEST_PARAMS['num_epochs']}")
print(f"早停耐心值: {BEST_PARAMS['patience']}")
print("=" * 80)



def evaluate_and_save_labeled_results(loader, filename_prefix, dataset_ids):
    """评估并保存有标签数据集的结果"""
    acc, true_labels, preds, probs = evaluate_labeled_model(best_model, loader)

    # 将准确率转换为小数形式
    acc_decimal = acc / 100

    metrics = calculate_metrics(true_labels, preds, probs)
    lower_bounds, upper_bounds = calculate_confidence_intervals(true_labels, preds, probs)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1 Score'],
        'Value': [acc_decimal] + list(metrics[1:]),
        'Lower Bound': lower_bounds,
        'Upper Bound': upper_bounds
    })
    metrics_df.to_csv(os.path.join(results_dir, f'{filename_prefix}_metrics.csv'), index=False)
    print(f'\n{filename_prefix} 结果:')
    print(metrics_df)

    # 将预测概率转换为数据框，每个类别对应一个列，并保留6位小数
    proba_df = pd.DataFrame(probs, columns=[f'Probability_{i}' for i in range(num_classes)])
    proba_df = proba_df.round(6)

    results_df = pd.DataFrame({
        'ID': dataset_ids,
        'True Label': true_labels,
        'Prediction': preds
    })
    results_df = pd.concat([results_df, proba_df], axis=1)

    # 设置浮点数格式，防止科学计数法
    results_df.to_csv(os.path.join(results_dir, f'{filename_prefix}_results.csv'),
                      index=False, float_format='%.6f')

    # 保存混淆矩阵
    cm = confusion_matrix(true_labels, preds)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(results_dir, f'{filename_prefix}_confusion_matrix.csv'), index=False)

    return {
        'Set': filename_prefix,
        'Accuracy': acc_decimal,
        'AUC': metrics[1],
        'Sensitivity': metrics[2],
        'Specificity': metrics[3],
        'Precision': metrics[4],
        'NPV': metrics[5],
        'F1 Score': metrics[6]
    }



def predict_and_save_unlabeled_results(loader, filename_prefix, dataset_ids):
    """预测并保存无标签数据集的结果"""
    all_ids, preds, probs = predict_unlabeled_model(best_model, loader)

    # 将预测概率转换为数据框
    proba_df = pd.DataFrame(probs, columns=[f'Probability_{i}' for i in range(num_classes)])
    proba_df = proba_df.round(6)

    # 创建结果数据框
    results_df = pd.DataFrame({
        'ID': all_ids,
        'Prediction': preds
    })


    if dataset_ids is not None:
        results_df['Original_ID'] = dataset_ids

    results_df = pd.concat([results_df, proba_df], axis=1)

    # 保存结果
    results_path = os.path.join(results_dir, f'{filename_prefix}_predictions.csv')
    results_df.to_csv(results_path, index=False, float_format='%.6f')

    print(f'\n{filename_prefix} 预测完成!')
    print(f'预测结果已保存到: {results_path}')
    print(f'样本数: {len(preds)}')
    print(f'类别分布: {pd.Series(preds).value_counts().to_dict()}')

    return {
        'Set': filename_prefix,
        'Sample_Count': len(preds),
        'Class_Distribution': pd.Series(preds).value_counts().to_dict()
    }


if __name__ == "__main__":
    # 设置随机种子
    seed = 109
    set_random_seed(seed)
    print(f"\n使用随机种子: {seed}")


    train_dataset = LabeledDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True, num_workers=0)


    inter_val_loader = DataLoader(LabeledDataset(X_inter_validation, Y_inter_validation),
                                  batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0)


    out_val_loader = DataLoader(LabeledDataset(X_out_validation, Y_out_validation),
                                batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0)


    unlabeled_loaders = {
        'Survival_Validation_1': DataLoader(UnlabeledDataset(X_survival_validation1, ids['survival_validation1']),
                                            batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0),
        'Survival_Validation_2': DataLoader(UnlabeledDataset(X_survival_validation2, ids['survival_validation2']),
                                            batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0),
        'Survival_Validation_3': DataLoader(UnlabeledDataset(X_survival_validation3, ids['survival_validation3']),
                                            batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0),
        'Immune_Validation_1': DataLoader(UnlabeledDataset(X_immune_validation1, ids['immune_validation1']),
                                          batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0),
        'Immune_Validation_2': DataLoader(UnlabeledDataset(X_immune_validation2, ids['immune_validation2']),
                                          batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=0)
    }

    # 初始化模型
    input_size = X_train.shape[1]
    num_classes = len(np.unique(Y_train))

    model = ImprovedNeuralNet(input_size, BEST_PARAMS['hidden_sizes'],
                              num_classes, BEST_PARAMS['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS['learning_rate'],
                           weight_decay=BEST_PARAMS['weight_decay'])

    # 记录开始时间
    start_time = time.time()

    print("\n开始训练...")
    best_model, best_val_accuracy, train_losses, val_accuracies = train_model_with_early_stopping(
        model, train_loader, inter_val_loader, BEST_PARAMS['num_epochs'],
        criterion, optimizer, BEST_PARAMS['patience']
    )

    # 保存训练过程
    training_history = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_accuracy': val_accuracies
    })
    training_history.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)


    model_save_path = os.path.join(results_dir, 'best_model_final.pth')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'params': BEST_PARAMS,
        'best_val_accuracy': best_val_accuracy,
        'random_seed': seed
    }, model_save_path)

    print(f"\n训练完成!")
    print(f"最佳内部验证集准确率: {best_val_accuracy:.2f}%")
    print(f"模型已保存到: {model_save_path}")
    print(f"总训练时间: {(time.time() - start_time) / 60:.1f} 分钟")

    # 评估和保存有标签数据集的结果
    print("\n" + "=" * 80)
    print("评估有标签数据集")
    print("=" * 80)

    all_results = []

    # 训练集
    all_results.append(evaluate_and_save_labeled_results(train_loader, 'Train_Set', ids['train_set']))

    # 内部验证集
    all_results.append(evaluate_and_save_labeled_results(inter_val_loader, 'Inter_Validation', ids['inter_validation']))

    # 外部验证集
    all_results.append(evaluate_and_save_labeled_results(out_val_loader, 'Out_Validation', ids['out_validation']))


    print("\n" + "=" * 80)
    print("预测无标签数据集")
    print("=" * 80)

    unlabeled_predictions = []

    for dataset_name, loader in unlabeled_loaders.items():
        predictions = predict_and_save_unlabeled_results(loader, dataset_name,
                                                         ids[dataset_name.lower().replace('_', '')])
        unlabeled_predictions.append(predictions)


    results_summary = pd.DataFrame(all_results)
    results_summary.to_csv(os.path.join(results_dir, 'labeled_datasets_metrics_summary.csv'), index=False)


    unlabeled_summary = pd.DataFrame(unlabeled_predictions)
    unlabeled_summary.to_csv(os.path.join(results_dir, 'unlabeled_datasets_predictions_summary.csv'), index=False)

    # 创建性能对比表格
    print("\n" + "=" * 80)
    print("最终模型性能汇总")
    print("=" * 80)


    print("\n有标签数据集性能:")
    for result in all_results:
        print(f"{result['Set']:20} | 准确率: {result['Accuracy']:.4f} | AUC: {result['AUC']:.4f} | "
              f"敏感性: {result['Sensitivity']:.4f} | 特异性: {result['Specificity']:.4f} | F1: {result['F1 Score']:.4f}")


    print("\n无标签数据集预测统计:")
    for prediction in unlabeled_predictions:
        print(
            f"{prediction['Set']:20} | 样本数: {prediction['Sample_Count']} | 类别分布: {prediction['Class_Distribution']}")

    print("\n" + "=" * 80)
    print(f"所有结果已保存到: {results_dir}")
    print("=" * 80)