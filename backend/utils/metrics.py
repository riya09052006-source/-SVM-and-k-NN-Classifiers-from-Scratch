import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split arrays X and y into random train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def cross_validation_split(X, y, folds=5, random_state=None):
    """Split dataset into k folds for cross-validation."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(folds, n_samples // folds)
    fold_sizes[:n_samples % folds] += 1
    current = 0
    splits = []
    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        splits.append((train_indices, val_indices))
        current += fold_size
    return splits

def accuracy_score(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, pos_label=1):
    """Calculate precision for binary classification: TP / (TP + FP)"""
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true, y_pred, pos_label=1):
    """Calculate recall for binary classification: TP / (TP + FN)"""
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true, y_pred, pos_label=1):
    """Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)"""
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix to evaluate classification accuracy."""
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1
    return cm, labels