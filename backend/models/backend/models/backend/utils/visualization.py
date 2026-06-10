import io
import base64
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg') # run without browser GUI windows

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_decision_boundary(model, X, y, title, filepath=None, return_base64=False):
    """Plot the decision boundary of a trained classifier on 2D data."""
    fig = plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    h = 0.04
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)
    
    cmap_light = plt.cm.get_cmap('RdYlBu', 2) if hasattr(plt.cm, 'get_cmap') else plt.colormaps['RdYlBu']
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', s=40, alpha=0.85)
    
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Feature 1', fontsize=10)
    plt.ylabel('Feature 2', fontsize=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    handles, labels = scatter.legend_elements()
    plt.legend(handles, ['Class 0 / -1', 'Class 1'], loc="upper right", frameon=True)
    plt.tight_layout()
    
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
        
    if filepath:
        plt.savefig(filepath, dpi=150)
    plt.close(fig)

def plot_elbow_curve(k_values, error_rates, optimal_k, filepath=None, return_base64=False):
    """Plot the Elbow Curve for k-NN hyperparameter selection."""
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(k_values, error_rates, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='CV Error Rate')
    
    try:
        opt_index = k_values.index(optimal_k)
        opt_error = error_rates[opt_index]
        plt.scatter(optimal_k, opt_error, color='#ff7f0e', s=120, zorder=5, edgecolor='k', label=f'Optimal k={optimal_k}')
        plt.annotate(f'Selected k={optimal_k}',
                     xy=(optimal_k, opt_error),
                     xytext=(optimal_k + 0.6, opt_error + 0.01),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=4),
                     fontsize=9, fontweight='bold')
    except Exception:
        pass
    
    plt.title('k-NN Elbow Curve (Cross-Validation Error)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Number of Neighbors (k)', fontsize=10)
    plt.ylabel('Error Rate', fontsize=10)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
        
    if filepath:
        plt.savefig(filepath, dpi=150)
    plt.close(fig)
    