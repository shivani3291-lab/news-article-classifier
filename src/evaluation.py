"""
Model evaluation utilities and metrics computation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, y_true, y_pred, model_name: str = "Model", 
                 class_names: list = None):
        """
        Initialize evaluator
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Name of the model for reporting
            class_names: List of class names (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.class_names = class_names or [f"Class {i}" for i in range(len(np.unique(y_true)))]
        
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Compute all performance metrics
        
        Returns:
            dict: Dictionary containing accuracy, precision, recall, F1-score
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
        }
        return metrics
    
    
    def print_report(self):
        """Print detailed classification report"""
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}\n")
        
        # Print metrics
        metrics = self.get_metrics()
        print("Overall Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_true, self.y_pred, 
                                   target_names=self.class_names))
    
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix
        
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (8, 6), 
                             cmap: str = 'Blues'):
        """
        Plot confusion matrix heatmap
        
        Args:
            figsize: Figure size
            cmap: Colormap
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics as DataFrame
        
        Returns:
            pd.DataFrame: Metrics in dataframe format
        """
        metrics = self.get_metrics()
        df = pd.DataFrame([metrics], index=[self.model_name])
        return df


def compare_models(models_dict: Dict[str, Tuple[Any, Any, str]], 
                   class_names: list = None) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models_dict: Dictionary of {model_name: (y_true, y_pred)}
        class_names: List of class names
        
    Returns:
        pd.DataFrame: Comparison table of all models
    """
    comparison_list = []
    
    for model_name, (y_true, y_pred) in models_dict.items():
        evaluator = ModelEvaluator(y_true, y_pred, model_name, class_names)
        metrics = evaluator.get_metrics()
        metrics['Model'] = model_name
        comparison_list.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_list)
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df.sort_values('f1', ascending=False)


def print_model_comparison(comparison_df: pd.DataFrame):
    """
    Print model comparison table nicely
    
    Args:
        comparison_df: Comparison dataframe from compare_models()
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")
    print(comparison_df.round(4).to_string())
    print("\n" + "="*80)
    print(f"Best Model: {comparison_df['f1'].idxmax()} (F1: {comparison_df['f1'].max():.4f})")
    print("="*80)


def get_per_class_metrics(y_true, y_pred, class_names: list = None) -> pd.DataFrame:
    """
    Get per-class performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        pd.DataFrame: Per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    return metrics_df


def plot_per_class_metrics(y_true, y_pred, class_names: list = None):
    """
    Plot per-class metrics as bar chart
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    metrics_df = get_per_class_metrics(y_true, y_pred, class_names)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Class'])
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_per_class_metrics(y_true, y_pred, class_names: list = None):
    """
    Print per-class metrics table
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    metrics_df = get_per_class_metrics(y_true, y_pred, class_names)
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE METRICS")
    print("="*70)
    print(metrics_df.round(4).to_string(index=False))
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Generate dummy data for demonstration
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 10)
    y_pred = np.array([0, 0, 1, 2, 2, 1, 3, 3] * 10)
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # Single model evaluation
    evaluator = ModelEvaluator(y_true, y_pred, "Demo Model", class_names)
    evaluator.print_report()
    evaluator.plot_confusion_matrix()
