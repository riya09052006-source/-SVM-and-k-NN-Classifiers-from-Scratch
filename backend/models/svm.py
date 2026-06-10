import numpy as np

class SVM:
    """
    Custom Support Vector Machine (SVM) Classifier.
    Solves the dual optimization problem using Sequential Minimal Optimization (SMO).
    """
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=None, coef0=1.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.original_classes = None
        
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        
    def _kernel_function(self, x1, x2):
        """Compute the kernel matrix between x1 and x2."""
        if self.kernel_type == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel_type == 'polynomial':
            return (np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel_type == 'rbf':
            sq_norm1 = np.sum(x1 ** 2, axis=1).reshape(-1, 1)
            sq_norm2 = np.sum(x2 ** 2, axis=1).reshape(1, -1)
            dists = sq_norm1 - 2 * np.dot(x1, x2.T) + sq_norm2
            return np.exp(-self.gamma * dists)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def fit(self, X, y):
        """Fit the SVM model using Platt's Simplified SMO algorithm."""
        self.X = np.asarray(X, dtype=float)
        self.y_raw = np.asarray(y)
        
        self.original_classes = np.unique(self.y_raw)
        if len(self.original_classes) != 2:
            raise ValueError("This SVM implementation only supports binary classification.")
            
        self.class_map = {self.original_classes[0]: -1, self.original_classes[1]: 1}
        self.rev_class_map = {-1: self.original_classes[0], 1: self.original_classes[1]}
        
        self.y = np.vectorize(self.class_map.get)(self.y_raw).astype(float)
        n_samples, n_features = self.X.shape
        
        if self.gamma is None:
            self.gamma = 1.0 / (n_features * self.X.var()) if self.X.var() > 0 else 1.0
            
        self.K = self._kernel_function(self.X, self.X)
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        passes = 0
        iteration = 0
        max_passes = 10
        
        while passes < max_passes and iteration < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                f_xi = np.sum(self.alpha * self.y * self.K[:, i]) + self.b
                E_i = f_xi - self.y[i]
                
                if ((self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or 
                    (self.y[i] * E_i > self.tol and self.alpha[i] > 0)):
                    
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    f_xj = np.sum(self.alpha * self.y * self.K[:, j]) + self.b
                    E_j = f_xj - self.y[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                        
                    if L == H:
                        continue
                        
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                        
                    new_alpha_j = self.alpha[j] - (self.y[j] * (E_i - E_j)) / eta
                    new_alpha_j = max(L, min(H, new_alpha_j))
                    
                    if abs(new_alpha_j - alpha_j_old) < 1e-5:
                        continue
                        
                    new_alpha_i = self.alpha[i] + self.y[i] * self.y[j] * (alpha_j_old - new_alpha_j)
                    self.alpha[i] = new_alpha_i
                    self.alpha[j] = new_alpha_j
                    
                    b1 = (self.b - E_i - 
                          self.y[i] * (new_alpha_i - alpha_i_old) * self.K[i, i] - 
                          self.y[j] * (new_alpha_j - alpha_j_old) * self.K[i, j])
                    
                    b2 = (self.b - E_j - 
                          self.y[i] * (new_alpha_i - alpha_i_old) * self.K[i, j] - 
                          self.y[j] * (new_alpha_j - alpha_j_old) * self.K[j, j])
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                        
                    num_changed_alphas += 1
            
            iteration += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
        sv_indices = self.alpha > 1e-5
        self.support_vectors = self.X[sv_indices]
        self.support_vector_labels = self.y[sv_indices]
        self.support_vector_alphas = self.alpha[sv_indices]
        
    def predict(self, X):
        """Predict labels for X."""
        X = np.asarray(X, dtype=float)
        if self.support_vectors is None or len(self.support_vectors) == 0:
            majority_class = self.original_classes[0]
            return np.full(X.shape[0], majority_class)
            
        K_sv = self._kernel_function(X, self.support_vectors)
        decisions = np.dot(K_sv, self.support_vector_alphas * self.support_vector_labels) + self.b
        predictions = np.sign(decisions)
        predictions[predictions == 0] = -1
        return np.vectorize(self.rev_class_map.get)(predictions)