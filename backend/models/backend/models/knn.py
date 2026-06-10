import numpy as np
from utils.metrics import cross_validation_split, accuracy_score

class KNN:
    """Custom k-Nearest Neighbors (k-NN) Classifier."""
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def _compute_distances(self, X):
        if self.metric == 'euclidean':
            sq_diff = np.sum(X**2, axis=1).reshape(-1, 1) - 2 * np.dot(X, self.X_train.T) + np.sum(self.X_train**2, axis=1).reshape(1, -1)
            sq_diff = np.clip(sq_diff, 0, None)
            return np.sqrt(sq_diff)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted.")
            
        dists = self._compute_distances(X)
        neighbors_indices = np.argsort(dists, axis=1)[:, :self.k]
        neighbors_labels = self.y_train[neighbors_indices]
        
        predictions = []
        for row_labels in neighbors_labels:
            unique_labels, counts = np.unique(row_labels, return_counts=True)
            best_idx = np.argmax(counts)
            predictions.append(unique_labels[best_idx])
        return np.array(predictions)

    @staticmethod
    def find_optimal_k(X, y, max_k=15, folds=5, metric='euclidean', random_state=42):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        actual_max_k = min(max_k, int(n_samples * (1 - 1/folds)) - 1)
        k_values = [k for k in range(1, actual_max_k + 1) if k % 2 != 0]
        
        if not k_values:
            k_values = [1]
            
        error_rates = []
        splits = cross_validation_split(X, y, folds=folds, random_state=random_state)
        
        for k in k_values:
            fold_errors = []
            for train_idx, val_idx in splits:
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                knn_clf = KNN(k=k, metric=metric)
                knn_clf.fit(X_train, y_train)
                preds = knn_clf.predict(X_val)
                
                error = 1.0 - accuracy_score(y_val, preds)
                fold_errors.append(error)
            error_rates.append(np.mean(fold_errors))
            
        if len(k_values) > 2:
            # Perpendicular distance elbow algorithm
            P1 = np.array([k_values[0], error_rates[0]])
            Pn = np.array([k_values[-1], error_rates[-1]])
            line_vec = Pn - P1
            line_len = np.linalg.norm(line_vec)
            
            distances = []
            for i in range(len(k_values)):
                Pi = np.array([k_values[i], error_rates[i]])
                d = np.abs(np.cross(Pn - P1, P1 - Pi)) / line_len if line_len > 0 else 0.0
                distances.append(d)
                
            optimal_k_idx = np.argmax(distances)
            optimal_k = k_values[optimal_k_idx]
        else:
            optimal_k = k_values[np.argmin(error_rates)]
            
        return optimal_k, k_values, error_rates