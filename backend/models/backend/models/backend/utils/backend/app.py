import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.svm import SVM
from models.knn import KNN
from utils.metrics import train_test_split, accuracy_score, precision_score, recall_score, f1_score
from utils.visualization import plot_decision_boundary, plot_elbow_curve

app = Flask(__name__)
CORS(app)

def generate_separable_data(n_samples=200, noise=0.15, random_state=42):
    np.random.seed(random_state)
    n_samples_half = n_samples // 2
    X0 = np.random.normal(loc=-1.2, scale=noise, size=(n_samples_half, 2))
    y0 = np.zeros(n_samples_half, dtype=int)
    X1 = np.random.normal(loc=1.2, scale=noise, size=(n_samples_half, 2))
    y1 = np.ones(n_samples_half, dtype=int)
    return np.vstack((X0, X1)), np.concatenate((y0, y1))

def generate_circles_data(n_samples=200, noise=0.08, factor=0.5, random_state=42):
    np.random.seed(random_state)
    n_samples_half = n_samples // 2
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    r_outer = 1.0 + np.random.normal(0, noise, n_samples_half)
    x_outer = r_outer * np.cos(angles[:n_samples_half])
    y_outer = r_outer * np.sin(angles[:n_samples_half])
    X_outer = np.column_stack((x_outer, y_outer))
    y_outer = np.zeros(n_samples_half, dtype=int)
    r_inner = factor + np.random.normal(0, noise, n_samples_half)
    x_inner = r_inner * np.cos(angles[n_samples_half:])
    y_inner = r_inner * np.sin(angles[n_samples_half:])
    X_inner = np.column_stack((x_inner, y_inner))
    y_inner = np.ones(n_samples_half, dtype=int)
    return np.vstack((X_outer, X_inner)), np.concatenate((y_outer, y_inner))

def generate_moons_data(n_samples=200, noise=0.15, random_state=42):
    np.random.seed(random_state)
    n_samples_half = n_samples // 2
    theta = np.linspace(0, np.pi, n_samples_half)
    x_outer = np.cos(theta) + np.random.normal(0, noise, n_samples_half)
    y_outer = np.sin(theta) + np.random.normal(0, noise, n_samples_half)
    X_outer = np.column_stack((x_outer, y_outer))
    y_outer = np.zeros(n_samples_half, dtype=int)
    x_inner = 1.0 - np.cos(theta) + np.random.normal(0, noise, n_samples_half)
    y_inner = 0.5 - np.sin(theta) + np.random.normal(0, noise, n_samples_half)
    X_inner = np.column_stack((x_inner, y_inner))
    y_inner = np.ones(n_samples_half, dtype=int)
    return np.vstack((X_outer, X_inner)), np.concatenate((y_outer, y_inner))

@app.route('/api/generate', methods=['POST'])
def generate_data():
    data = request.get_json() or {}
    dataset_type = data.get('dataset_type', 'moons')
    samples = int(data.get('samples', 200))
    noise = float(data.get('noise', 0.15))
    
    if dataset_type == 'separable':
        X, y = generate_separable_data(n_samples=samples, noise=noise)
    elif dataset_type == 'circles':
        X, y = generate_circles_data(n_samples=samples, noise=noise)
    else:
        X, y = generate_moons_data(n_samples=samples, noise=noise)
        
    return jsonify({'X': X.tolist(), 'y': y.tolist()})

@app.route('/api/train_knn', methods=['POST'])
def train_knn():
    data = request.get_json() or {}
    X = np.array(data.get('X'))
    y = np.array(data.get('y'))
    k = int(data.get('k', 5))
    metric = data.get('metric', 'euclidean')
    auto_k = bool(data.get('auto_k', False))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    elbow_curve_img = None
    if auto_k:
        k, k_values, error_rates = KNN.find_optimal_k(X_train, y_train, max_k=15, folds=5, metric=metric)
        elbow_curve_img = plot_elbow_curve(k_values, error_rates, k, return_base64=True)
        
    knn = KNN(k=k, metric=metric)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    boundary_title = f"k-NN Decision Boundary (k={k}, Metric={metric.capitalize()})"
    decision_boundary_img = plot_decision_boundary(knn, X_train, y_train, title=boundary_title, return_base64=True)
    
    return jsonify({
        'k': k,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'decision_boundary_img': decision_boundary_img,
        'elbow_curve_img': elbow_curve_img
    })

@app.route('/api/train_svm', methods=['POST'])
def train_svm():
    data = request.get_json() or {}
    X = np.array(data.get('X'))
    y = np.array(data.get('y'))
    C = float(data.get('C', 1.0))
    kernel = data.get('kernel', 'linear')
    degree = int(data.get('degree', 3))
    gamma = data.get('gamma')
    
    if gamma is not None:
        gamma = float(gamma)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=1500)
    
    try:
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
        
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        gamma_label = f"{svm.gamma:.3f}" if svm.gamma else "Auto"
        boundary_title = f"SVM Decision Boundary ({kernel.capitalize()} Kernel, C={C}, Gamma={gamma_label})"
        decision_boundary_img = plot_decision_boundary(svm, X_train, y_train, title=boundary_title, return_base64=True)
        
        return jsonify({
            'success': True,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'decision_boundary_img': decision_boundary_img
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)