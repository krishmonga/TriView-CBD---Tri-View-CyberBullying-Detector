"""
Enhanced Visualization and Analysis Utilities for IEEE Paper
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from typing import List, Dict, Tuple, Any
import torch
import json
import os
from datetime import datetime
import pandas as pd

def set_plot_style():
    """Set consistent plot style for IEEE papers"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def plot_training_history(history: Dict, save_path: str = None, show: bool = True):
    """Plot comprehensive training history"""
    set_plot_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    if 'train_acc' in history and 'val_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning rate schedule
    if 'learning_rates' in history:
        epochs = range(1, len(history['learning_rates']) + 1)
        axes[0, 2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Attention entropy
    if 'attention_entropy' in history:
        epochs = range(1, len(history['attention_entropy']) + 1)
        axes[1, 0].plot(epochs, history['attention_entropy'], 'purple', linewidth=2)
        axes[1, 0].set_title('Attention Entropy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. View attention weights
    if 'view_attention_weights' in history:
        epochs = range(1, len(next(iter(history['view_attention_weights'].values()))) + 1)
        view_colors = {'Lexical': '#1f77b4', 'Semantic': '#2ca02c', 'Structural': '#d62728'}
        
        for view_name, weights in history['view_attention_weights'].items():
            axes[1, 1].plot(epochs, weights, label=view_name, 
                           color=view_colors.get(view_name), linewidth=2)
        
        axes[1, 1].set_title('View Attention Weights', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. F1 Score if available
    if 'train_f1' in history and 'val_f1' in history:
        epochs = range(1, len(history['train_f1']) + 1)
        axes[1, 2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[1, 2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
        axes[1, 2].set_title('Training and Validation F1-Score', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix_with_metrics(y_true, y_pred, save_path: str = None,
                                     class_names: List[str] = None,
                                     normalize: bool = False, show: bool = True):
    """Plot confusion matrix with detailed metrics"""
    set_plot_style()
    
    if class_names is None:
        class_names = ['Non-Bullying', 'Bullying']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=axes[0])
    
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # 2. Metrics bar plot
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = axes[1].bar(x - width, precision, width, label='Precision', 
                       color='#2ca02c', edgecolor='black')
    bars2 = axes[1].bar(x, recall, width, label='Recall', 
                       color='#1f77b4', edgecolor='black')
    bars3 = axes[1].bar(x + width, f1, width, label='F1-Score', 
                       color='#d62728', edgecolor='black')
    
    axes[1].set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].legend()
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print detailed classification report
    print("\n📋 Detailed Classification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    return cm

def plot_roc_curve(y_true, y_prob, save_path: str = None, show: bool = True):
    """Plot ROC curve"""
    set_plot_style()
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_prob, save_path: str = None, show: bool = True):
    """Plot Precision-Recall curve"""
    set_plot_style()
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Precision-Recall curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return avg_precision

def plot_attention_analysis(attention_analysis: Dict, save_path: str = None, show: bool = True):
    """Plot comprehensive attention analysis"""
    if not attention_analysis:
        print("⚠ No attention analysis data available")
        return
    
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean attention weights bar plot
    if 'mean_weights' in attention_analysis:
        views = list(attention_analysis['mean_weights'].keys())
        mean_values = [attention_analysis['mean_weights'][view] for view in views]
        std_values = [attention_analysis.get('std_weights', {}).get(view, 0) for view in views]
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        x_pos = np.arange(len(views))
        
        bars = axes[0, 0].bar(x_pos, mean_values, yerr=std_values, 
                             capsize=5, color=colors, edgecolor='black', alpha=0.8)
        axes[0, 0].set_title('Mean Attention Weights by View', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('View')
        axes[0, 0].set_ylabel('Attention Weight')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(views)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars, mean_values, std_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{mean_val:.3f} ± {std_val:.3f}', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Dominant view distribution
    if 'dominant_view_distribution' in attention_analysis:
        views = list(attention_analysis['dominant_view_distribution'].keys())
        percentages = [attention_analysis['dominant_view_distribution'][view]['percentage'] 
                      for view in views]
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        wedges, texts, autotexts = axes[0, 1].pie(
            percentages, labels=views, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=[0.05]*len(views), shadow=True
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        axes[0, 1].set_title('Dominant View Distribution', fontsize=12, fontweight='bold')
    
    # 3. Attention metrics radar chart
    if 'attention_metrics' in attention_analysis:
        metrics = ['Entropy', 'Diversity', 'Sparsity']
        values = [
            attention_analysis['attention_metrics'].get('entropy', 0),
            attention_analysis['attention_metrics'].get('diversity', 0),
            attention_analysis['attention_metrics'].get('sparsity', 0)
        ]
        
        # Normalize for radar chart
        max_vals = [2.0, 1.0, 3.0]  # Max expected values
        normalized = [v/max_v for v, max_v in zip(values, max_vals)]
        
        # Complete the circle
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        normalized += normalized[:1]
        angles += angles[:1]
        
        axes[1, 0] = plt.subplot(2, 2, 3, polar=True)
        axes[1, 0].plot(angles, normalized, 'o-', linewidth=2, color='#9467bd')
        axes[1, 0].fill(angles, normalized, alpha=0.25, color='#9467bd')
        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].set_title('Attention Metrics Radar Chart', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True)
    
    # 4. Text length vs attention correlation
    if 'text_attention_correlations' in attention_analysis:
        views = list(attention_analysis['text_attention_correlations'].keys())
        correlations = [attention_analysis['text_attention_correlations'][view] 
                       for view in views]
        
        colors = ['#1f77b4' if c >= 0 else '#d62728' for c in correlations]
        bars = axes[1, 1].bar(views, correlations, color=colors, edgecolor='black', alpha=0.8)
        
        axes[1, 1].set_title('Text Length vs Attention Correlation', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('View')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add correlation value labels
        for bar, corr in zip(bars, correlations):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.03),
                           f'{corr:.3f}', 
                           ha='center', va='bottom' if corr >= 0 else 'top',
                           fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Attention analysis plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_model_comparison(results: Dict, save_path: str = None, show: bool = True):
    """Plot model comparison results"""
    set_plot_style()
    
    models = list(results.keys())
    
    # Extract metrics
    accuracies = [results[m].get('accuracy', 0) for m in models]
    f1_scores = [results[m].get('f1_score', 0) for m in models]
    precisions = [results[m].get('precision', 0) for m in models]
    recalls = [results[m].get('recall', 0) for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy comparison
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', 
                  color='#1f77b4', edgecolor='black', alpha=0.8)
    axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', 
                  color='#2ca02c', edgecolor='black', alpha=0.8)
    
    axes[0, 0].set_title('Model Accuracy and F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.replace('_', '\n').title() for m in models], 
                              rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0.8, 1.0])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Precision-Recall comparison
    x_pos = np.arange(len(models))
    width = 0.35
    
    axes[0, 1].bar(x_pos - width/2, precisions, width, label='Precision', 
                  color='#ff7f0e', edgecolor='black', alpha=0.8)
    axes[0, 1].bar(x_pos + width/2, recalls, width, label='Recall', 
                  color='#d62728', edgecolor='black', alpha=0.8)
    
    axes[0, 1].set_title('Model Precision and Recall Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([m.replace('_', '\n').title() for m in models], 
                              rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0.8, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Performance ranking (sorted by accuracy)
    sorted_models = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    sorted_names = [m[0].replace('_', ' ').title() for m in sorted_models]
    sorted_acc = [m[1] for m in sorted_models]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))
    bars = axes[1, 0].barh(range(len(sorted_models)), sorted_acc, color=colors, edgecolor='black')
    
    axes[1, 0].set_title('Model Performance Ranking (by Accuracy)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_ylabel('Model')
    axes[1, 0].set_yticks(range(len(sorted_models)))
    axes[1, 0].set_yticklabels(sorted_names)
    axes[1, 0].set_xlim([0.8, 1.0])
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Add accuracy labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        axes[1, 0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{acc:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # 4. Improvement over baseline (if baseline exists)
    if 'bilstm' in results or 'cnn' in results:
        baseline_keys = [k for k in ['bilstm', 'cnn'] if k in results]
        if baseline_keys:
            baseline_acc = np.mean([results[k]['accuracy'] for k in baseline_keys])
            improvements = {}
            
            for model in models:
                if model not in baseline_keys:
                    improvements[model] = (results[model]['accuracy'] - baseline_acc) / baseline_acc * 100
            
            if improvements:
                imp_models = list(improvements.keys())
                imp_values = [improvements[m] for m in imp_models]
                
                colors = ['#2ca02c' if v > 0 else '#d62728' for v in imp_values]
                bars = axes[1, 1].bar(imp_models, imp_values, color=colors, edgecolor='black')
                
                axes[1, 1].set_title('Improvement Over Baseline (%)', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Model')
                axes[1, 1].set_ylabel('Improvement (%)')
                axes[1, 1].set_xticklabels([m.replace('_', '\n').title() for m in imp_models], 
                                          rotation=45, ha='right')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                # Add improvement labels
                for bar, imp in zip(bars, imp_values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                                   bar.get_height() + (0.5 if imp > 0 else -1),
                                   f'{imp:.1f}%', 
                                   ha='center', va='bottom' if imp > 0 else 'top',
                                   fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def generate_latex_tables(results: Dict, output_dir: str):
    """Generate LaTeX tables for IEEE paper"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: Main results comparison
    table1 = """\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Different Models}
\\label{tab:model-comparison}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\midrule
"""
    
    # Sort models by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, metrics in sorted_models:
        display_name = model_name.replace('_', ' ').title()
        table1 += f"{display_name} & {metrics['accuracy']:.4f} & {metrics['f1_score']:.4f} & {metrics['precision']:.4f} & {metrics['recall']:.4f} \\\\\n"
    
    table1 += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    with open(os.path.join(output_dir, 'table1_model_comparison.tex'), 'w') as f:
        f.write(table1)
    
    # Table 2: Ablation study results
    ablation_models = [m for m in results.keys() if 'only' in m or 'attention' in m or 'lexical_' in m or 'semantic_' in m]
    
    if ablation_models:
        table2 = """\\begin{table}[htbp]
\\centering
\\caption{Ablation Study Results}
\\label{tab:ablation-study}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Model Variant} & \\textbf{Accuracy} & \\textbf{F1-Score} \\\\
\\midrule
"""
        
        for model_name in ablation_models:
            if model_name in results:
                display_name = model_name.replace('_', ' ').title()
                table2 += f"{display_name} & {results[model_name]['accuracy']:.4f} & {results[model_name]['f1_score']:.4f} \\\\\n"
        
        table2 += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(os.path.join(output_dir, 'table2_ablation_study.tex'), 'w') as f:
            f.write(table2)
    
    # Table 3: Attention analysis
    print("📄 LaTeX tables generated in:", output_dir)

def save_results_to_csv(results: Dict, output_path: str):
    """Save results to CSV file"""
    data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1_Score': f"{metrics['f1_score']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False)
    
    df.to_csv(output_path, index=False)
    print(f"📊 Results saved to CSV: {output_path}")
    
    return df

def create_comprehensive_report(results: Dict, history: Dict = None, 
                               output_dir: str = 'outputs'):
    """Create comprehensive report with all visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f'report_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n📝 Creating comprehensive report in: {report_dir}")
    
    # 1. Save raw results
    with open(os.path.join(report_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 2. Save training history if available
    if history:
        with open(os.path.join(report_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Plot training history
        plot_training_history(
            history,
            save_path=os.path.join(report_dir, 'training_history.png'),
            show=False
        )
    
    # 3. Model comparison plot
    plot_model_comparison(
        results,
        save_path=os.path.join(report_dir, 'model_comparison.png')
    )
    
    # 4. Generate LaTeX tables
    generate_latex_tables(results, os.path.join(report_dir, 'latex_tables'))
    
    # 5. Save to CSV
    csv_path = os.path.join(report_dir, 'results.csv')
    df = save_results_to_csv(results, csv_path)
    
    # 6. Create summary text file
    summary_path = os.path.join(report_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRI-FUSE CYBERBULLYING DETECTION - EXPERIMENT REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}\n")
        f.write("-" * 70 + "\n")
        
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            display_name = model_name.replace('_', ' ').title()
            f.write(f"{display_name:<25} {metrics['accuracy']:.4f}       {metrics['f1_score']:.4f}       "
                   f"{metrics['precision']:.4f}       {metrics['recall']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("=" * 70 + "\n\n")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        worst_model = min(results.items(), key=lambda x: x[1]['accuracy'])
        
        f.write(f"1. Best Performing Model: {best_model[0].replace('_', ' ').title()}\n")
        f.write(f"   Accuracy: {best_model[1]['accuracy']:.4f}\n\n")
        
        f.write(f"2. Worst Performing Model: {worst_model[0].replace('_', ' ').title()}\n")
        f.write(f"   Accuracy: {worst_model[1]['accuracy']:.4f}\n\n")
        
        # Check TriFuse performance
        if 'trifuse' in results:
            trifuse_acc = results['trifuse']['accuracy']
            if trifuse_acc >= 0.95:
                f.write("3. ✅ SUCCESS: TriFuse achieved target accuracy of 95%+\n")
            else:
                f.write(f"3. ⚠ TriFuse accuracy: {trifuse_acc:.4f} (below 95% target)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("=" * 70 + "\n")
        f.write(f"1. results.json - Complete results in JSON format\n")
        f.write(f"2. results.csv - Results in CSV format\n")
        f.write(f"3. training_history.png - Training curves\n")
        f.write(f"4. model_comparison.png - Model comparison chart\n")
        f.write(f"5. latex_tables/ - LaTeX tables for IEEE paper\n")
        f.write(f"6. summary.txt - This summary file\n")
    
    print(f"✅ Report created successfully in: {report_dir}")
    print(f"📄 Summary file: {summary_path}")
    
    # Print summary to console
    print("\n📋 REPORT SUMMARY:")
    print("-" * 50)
    
    # Top 3 models
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
    for i, (model, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {model.replace('_', ' ').title():<20}: {metrics['accuracy']:.4f}")
    
    return report_dir

def analyze_attention_patterns_by_class(attention_weights: torch.Tensor, 
                                       labels: torch.Tensor,
                                       predictions: torch.Tensor,
                                       view_names: List[str] = None):
    """Analyze attention patterns by class and prediction correctness"""
    if view_names is None:
        view_names = ['Lexical', 'Semantic', 'Structural']
    
    attention_np = attention_weights.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np = predictions.cpu().numpy()
    
    analysis = {
        'by_true_class': {},
        'by_prediction_correctness': {},
        'confusion_matrix_attention': {}
    }
    
    # Analyze by true class
    unique_labels = np.unique(labels_np)
    for label in unique_labels:
        mask = labels_np == label
        label_attention = attention_np[mask]
        
        if len(label_attention) > 0:
            analysis['by_true_class'][f'class_{label}'] = {
                'samples': int(mask.sum()),
                'mean_attention': {
                    view_names[i]: float(np.mean(label_attention[:, i]))
                    for i in range(len(view_names))
                }
            }
    
    # Analyze by prediction correctness
    correct_mask = preds_np == labels_np
    incorrect_mask = preds_np != labels_np
    
    if correct_mask.any():
        correct_attention = attention_np[correct_mask]
        analysis['by_prediction_correctness']['correct'] = {
            'samples': int(correct_mask.sum()),
            'mean_attention': {
                view_names[i]: float(np.mean(correct_attention[:, i]))
                for i in range(len(view_names))
            }
        }
    
    if incorrect_mask.any():
        incorrect_attention = attention_np[incorrect_mask]
        analysis['by_prediction_correctness']['incorrect'] = {
            'samples': int(incorrect_mask.sum()),
            'mean_attention': {
                view_names[i]: float(np.mean(incorrect_attention[:, i]))
                for i in range(len(view_names))
            }
        }
    
    return analysis