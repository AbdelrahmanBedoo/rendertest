"""
Visualization functions for Logistic Regression model results.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


def visualize_expanding_window_performance(trained: Dict, tickers: List[str]) -> None:
    """
    Comprehensive visualization for expanding window performance.
    Shows cumulative accuracy, precision, recall, and ROC curves.
    Each graph is displayed in a separate window for better clarity.
    """
    n_tickers = len(tickers)
    if n_tickers == 0:
        return
    
    print(f"\n{'='*80}")
    print(f"EXPANDING WINDOW PERFORMANCE OVER TIME")
    print(f"{'='*80}")
    print(f"  Tickers with expanding window: {', '.join(tickers)}")
    
    # Set style for better-looking plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('seaborn')
    
    for ticker_idx, ticker in enumerate(tickers):
        res = trained[ticker]
        if 'all_predictions_list' not in res or not res['all_predictions_list']:
            continue
        
        predictions = res['all_predictions_list']
        n_predictions = len(predictions)
        
        if n_predictions == 0:
            continue
        
        # Extract actuals and predictions
        all_actuals = res['y_test_actual']
        all_predictions = np.array([p['pred'] for p in predictions])
        all_probas = res['y_test_proba']
        
        # Calculate cumulative metrics as we progress through predictions
        cumulative_tp = 0  # True Positives (UP predicted correctly)
        cumulative_fp = 0  # False Positives (UP predicted but actually DOWN)
        cumulative_tn = 0  # True Negatives (DOWN predicted correctly)
        cumulative_fn = 0  # False Negatives (DOWN predicted but actually UP)
        
        cumulative_acc = []
        cumulative_precision_up = []    # Precision for UP class
        cumulative_recall_up = []       # Recall for UP class
        cumulative_precision_down = []  # Precision for DOWN class
        cumulative_recall_down = []     # Recall for DOWN class
        window_indices = []
        
        for idx, pred in enumerate(predictions):
            # Update confusion matrix counts
            if pred['type'] == 'TP':
                cumulative_tp += 1
            elif pred['type'] == 'FP':
                cumulative_fp += 1
            elif pred['type'] == 'TN':
                cumulative_tn += 1
            else:  # FN
                cumulative_fn += 1
            
            # Calculate cumulative metrics
            total_so_far = cumulative_tp + cumulative_fp + cumulative_tn + cumulative_fn
            if total_so_far > 0:
                acc = (cumulative_tp + cumulative_tn) / total_so_far
                cumulative_acc.append(acc)
                
                # Precision and Recall for UP class
                if cumulative_tp + cumulative_fp > 0:
                    prec_up = cumulative_tp / (cumulative_tp + cumulative_fp)
                else:
                    prec_up = 0.0
                cumulative_precision_up.append(prec_up)
                
                if cumulative_tp + cumulative_fn > 0:
                    rec_up = cumulative_tp / (cumulative_tp + cumulative_fn)
                else:
                    rec_up = 0.0
                cumulative_recall_up.append(rec_up)
                
                # Precision and Recall for DOWN class
                if cumulative_tn + cumulative_fn > 0:
                    prec_down = cumulative_tn / (cumulative_tn + cumulative_fn)
                else:
                    prec_down = 0.0
                cumulative_precision_down.append(prec_down)
                
                if cumulative_tn + cumulative_fp > 0:
                    rec_down = cumulative_tn / (cumulative_tn + cumulative_fp)
                else:
                    rec_down = 0.0
                cumulative_recall_down.append(rec_down)
                
                window_indices.append(idx + 1)
        
        # Get overall metrics
        eval_dict = res['eval_baseline']
        
        # ========================================================================
        # PLOT 1: Cumulative Metrics Over Time (Separate Window)
        # ========================================================================
        fig1, ax1 = plt.subplots(figsize=(16, 9))
        fig1.patch.set_facecolor('white')
        
        # Plot lines with better styling - showing all metrics
        ax1.plot(window_indices, cumulative_acc, 'o-', label='Accuracy', 
                color='#2E86AB', linewidth=3, markersize=4, alpha=0.9, 
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB', zorder=6)
        
        # UP class metrics
        ax1.plot(window_indices, cumulative_precision_up, 's-', label='Precision (UP)', 
                color='#06A77D', linewidth=2.5, markersize=4, alpha=0.85,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#06A77D', zorder=5)
        ax1.plot(window_indices, cumulative_recall_up, '^-', label='Recall (UP)', 
                color='#F24236', linewidth=2.5, markersize=4, alpha=0.85,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#F24236', zorder=4)
        
        # DOWN class metrics
        ax1.plot(window_indices, cumulative_precision_down, 'D-', label='Precision (DOWN)', 
                color='#7209B7', linewidth=2.5, markersize=4, alpha=0.85,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#7209B7', zorder=3)
        ax1.plot(window_indices, cumulative_recall_down, 'v-', label='Recall (DOWN)', 
                color='#F77F00', linewidth=2.5, markersize=4, alpha=0.85,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#F77F00', zorder=2)
        
        ax1.axhline(y=0.5, color='#666666', linestyle='--', linewidth=2, 
                   alpha=0.6, label='Random Baseline (0.5)', zorder=0)
        
        # Add final values as styled text box
        if cumulative_acc:
            final_acc = cumulative_acc[-1]
            final_prec_up = cumulative_precision_up[-1]
            final_rec_up = cumulative_recall_up[-1]
            final_prec_down = cumulative_precision_down[-1]
            final_rec_down = cumulative_recall_down[-1]
            textstr = (f'Final Metrics:\n\n'
                      f'Accuracy:        {final_acc:.4f}\n'
                      f'Precision (UP):  {final_prec_up:.4f}\n'
                      f'Recall (UP):     {final_rec_up:.4f}\n'
                      f'Precision (DOWN): {final_prec_down:.4f}\n'
                      f'Recall (DOWN):   {final_rec_down:.4f}')
            ax1.text(0.98, 0.02, textstr, 
                   transform=ax1.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=1', facecolor='#FFF9E3', 
                           edgecolor='#333333', linewidth=2, alpha=0.9))
        
        ax1.set_xlabel('Number of Predictions', fontweight='bold', fontsize=14, labelpad=10)
        ax1.set_ylabel('Score', fontweight='bold', fontsize=14, labelpad=10)
        ax1.set_title(f'{ticker} - Cumulative Metrics Over Time (Expanding Window)', 
                     fontweight='bold', fontsize=16, pad=20)
        ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        ax1.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True, 
                  fancybox=True, edgecolor='gray', ncol=2)
        ax1.set_ylim([0, 1.05])
        ax1.set_xlim([0, max(window_indices) if window_indices else 1])
        
        # Style the plot
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        
        plt.tight_layout()
        plt.show()
        
        # ========================================================================
        # PLOT 2: ROC Curve (Separate Window)
        # ========================================================================
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        fig2.patch.set_facecolor('white')
        
        fpr, tpr, _ = roc_curve(all_actuals, all_probas)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve with gradient fill
        ax2.plot(fpr, tpr, lw=4, label=f'ROC Curve (AUC = {roc_auc:.4f})', 
                color='#2E86AB', alpha=0.9)
        ax2.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
        ax2.plot([0, 1], [0, 1], color='#666666', lw=2, linestyle='--', 
                alpha=0.7, label='Random Classifier', zorder=0)
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=14, labelpad=10)
        ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=14, labelpad=10)
        ax2.set_title(f'{ticker} - Receiver Operating Characteristic (ROC) Curve', 
                     fontweight='bold', fontsize=16, pad=20)
        ax2.legend(loc='lower right', fontsize=13, framealpha=0.95, shadow=True,
                  fancybox=True, edgecolor='gray')
        ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        
        # Add AUC text box
        ax2.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', 
                transform=ax2.transAxes, fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='#FFF9E3', 
                         edgecolor='#333333', linewidth=2, alpha=0.9))
        
        # Style the plot
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(1.5)
        ax2.spines['bottom'].set_linewidth(1.5)
        ax2.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        
        plt.tight_layout()
        plt.show()
        
        # ========================================================================
        # PLOT 3: Confusion Matrix (Separate Window)
        # ========================================================================
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        fig3.patch.set_facecolor('white')
        
        cm = confusion_matrix(all_actuals, all_predictions)
        
        # Create a more visually appealing confusion matrix
        im = ax3.imshow(cm, cmap='Blues', alpha=0.8, aspect='auto', vmin=0, 
                       vmax=cm.max() * 1.2)
        
        # Add text annotations with better styling
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax3.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                        fontsize=28, fontweight='bold', color=text_color)
        
        # Calculate percentages
        total = cm.sum()
        percentages = (cm / total * 100).round(1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax3.text(j, i + 0.35, f'({percentages[i, j]:.1f}%)', 
                        ha='center', va='center', fontsize=14, 
                        fontweight='bold', color=text_color)
        
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Predicted DOWN', 'Predicted UP'], 
                           fontsize=13, fontweight='bold')
        ax3.set_yticklabels(['Actual DOWN', 'Actual UP'], 
                           fontsize=13, fontweight='bold')
        
        # Add colorbar with better styling
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Count', fontsize=12, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=11)
        
        ax3.set_title(f'{ticker} - Confusion Matrix', 
                     fontweight='bold', fontsize=16, pad=20)
        
        # Style the plot
        for spine in ax3.spines.values():
            spine.set_linewidth(2)
            spine.set_color('#333333')
        
        plt.tight_layout()
        plt.show()
        
        # ========================================================================
        # PLOT 4: Overall Performance Metrics Bar Chart (Separate Window)
        # ========================================================================
        fig4, ax4 = plt.subplots(figsize=(14, 8))
        fig4.patch.set_facecolor('white')
        
        # Extract per-class metrics
        precision_per_class = eval_dict.get('precision_per_class', [0.0, 0.0])
        recall_per_class = eval_dict.get('recall_per_class', [0.0, 0.0])
        f1_per_class = eval_dict.get('f1_per_class', [0.0, 0.0])
        
        # Ensure we have both classes (DOWN=0, UP=1)
        if len(precision_per_class) < 2:
            precision_per_class = precision_per_class + [0.0] * (2 - len(precision_per_class))
        if len(recall_per_class) < 2:
            recall_per_class = recall_per_class + [0.0] * (2 - len(recall_per_class))
        if len(f1_per_class) < 2:
            f1_per_class = f1_per_class + [0.0] * (2 - len(f1_per_class))
        
        metrics = ['Accuracy', 'Precision\n(UP)', 'Precision\n(DOWN)', 
                  'Recall\n(UP)', 'Recall\n(DOWN)', 'F1-Score\n(UP)', 
                  'F1-Score\n(DOWN)', 'AUC']
        values = [
            eval_dict['accuracy'],
            precision_per_class[1],  # UP class
            precision_per_class[0],  # DOWN class
            recall_per_class[1],     # UP class
            recall_per_class[0],     # DOWN class
            f1_per_class[1],         # UP class
            f1_per_class[0],         # DOWN class
            eval_dict['auc']
        ]
        
        # Use gradient colors - matching the line plot colors
        colors = ['#2E86AB', '#06A77D', '#7209B7', '#F24236', '#F77F00', 
                 '#06A77D', '#7209B7', '#2E86AB']
        bars = ax4.bar(metrics, values, alpha=0.85, color=colors, 
                      edgecolor='#333333', linewidth=2.5, 
                      width=0.7, zorder=3)
        
        # Add value labels on bars with better styling
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                   f'{val:.4f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        
        # Add baseline reference line
        ax4.axhline(y=0.5, color='#666666', linestyle='--', linewidth=2.5, 
                   alpha=0.7, label='Baseline (0.5)', zorder=1)
        
        # Add performance zone shading
        ax4.axhspan(0.7, 1.0, alpha=0.1, color='green', label='Excellent (>0.7)', zorder=0)
        ax4.axhspan(0.5, 0.7, alpha=0.1, color='yellow', label='Good (0.5-0.7)', zorder=0)
        
        ax4.set_ylabel('Score', fontweight='bold', fontsize=14, labelpad=10)
        ax4.set_title(f'{ticker} - Overall Performance Metrics Summary (Per-Class)', 
                     fontweight='bold', fontsize=16, pad=20)
        ax4.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.8, zorder=0)
        ax4.set_ylim([0, 1.15])
        ax4.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                  shadow=True, fancybox=True, edgecolor='gray')
        
        # Style x-axis labels - rotate for better readability
        ax4.set_xticklabels(metrics, fontsize=11, fontweight='bold', rotation=15, ha='right')
        ax4.tick_params(axis='x', which='major', pad=10)
        
        # Style the plot
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.spines['bottom'].set_linewidth(1.5)
        ax4.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n✓ Visualizations completed for {ticker}")


def visualize_threshold_optimization(trained: Dict, regular_tickers: List[str]) -> None:
    """Visualize threshold optimization results for regular (non-expanding window) tickers."""
    if not regular_tickers:
        return
    
    cols = 2
    rows = int(np.ceil(len(regular_tickers)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
    fig.suptitle('Threshold Optimization Results (Validation Set)', fontsize=16, fontweight='bold')
    
    for idx, ticker in enumerate(regular_tickers):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col] if len(regular_tickers) > 1 else axes
        
        res = trained[ticker]
        if res.get('f1_scores'):
            ax.plot(res['thresholds'], res['f1_scores'], linewidth=2, label='F1 Score (Validation)')
            ax.scatter([res['best_threshold']], [res.get('best_val_f1', 0.0)], 
                      color='red', s=120, marker='*', 
                      label=f"Optimal: {res['best_threshold']:.3f}")
            ax.axvline(x=res['best_threshold'], color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        else:
            ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score (UP class)')
        bp = res['best_params']
        ax.set_title(f"{ticker} (C={bp['C']:.2f}, max_iter={bp['max_iter']})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0, 1])
        ax.legend()
    
    # Hide unused subplots
    total_plots = rows * cols
    for extra_idx in range(len(regular_tickers), total_plots):
        row = extra_idx // cols
        col = extra_idx % cols
        ax_to_hide = axes[row, col] if rows > 1 else axes[col] if len(regular_tickers) > 1 else None
        if ax_to_hide:
            ax_to_hide.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_performance_metrics(comparison_df: pd.DataFrame) -> None:
    """Visualize performance metrics comparison across stocks."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Cross-Stock Performance Metrics Comparison', 
                 fontsize=16, fontweight='bold')
    
    metrics_config = [
        ('Test Acc (0.5)', 'Test Accuracy\n(Baseline 0.5)', 'blue'),
        ('Test Acc (Used)', 'Test Accuracy\n(Threshold Used)', 'orange'),
        ('Test Acc Δ', 'Test Accuracy\nImprovement', 'green'),
        ('Test F1-UP (0.5)', 'Test F1-Score (UP)\n(Baseline 0.5)', 'blue'),
        ('Test F1-UP (Used)', 'Test F1-Score (UP)\n(Threshold Used)', 'orange'),
        ('Test F1 Δ', 'Test F1-Score\nImprovement', 'green')
    ]
    
    for idx, (metric, name, default_color) in enumerate(metrics_config):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = comparison_df[metric].values
        bars = ax.bar(comparison_df['Stock'].values, values, alpha=0.8, color=default_color, edgecolor='black', linewidth=1.2)
        ax.set_title(name, fontsize=11, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars based on improvement metrics (green for positive, red for negative)
        if 'Δ' in metric:
            for bar, val in zip(bars, values):
                bar.set_color('green' if val > 0 else ('red' if val < 0 else 'gray'))
            # Add horizontal line at 0 for improvement metrics
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars  
        for bar in bars:
            height = bar.get_height()
            va_pos = 'bottom' if height >= 0 else 'top'
            y_pos = height + (0.01 if height >= 0 else -0.01)
            ax.text(bar.get_x() + bar.get_width()/2., y_pos, 
                   f'{height:.3f}', ha='center', va=va_pos, fontsize=9, fontweight='bold')
        
        # Set y-axis limits appropriately
        if 'Δ' in metric:
            y_range = values.max() - values.min()
            y_margin = max(0.05, y_range * 0.1)
            ax.set_ylim([values.min() - y_margin, values.max() + y_margin])
        else:
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def visualize_roc_curves(trained: Dict, past_window: int, future_window: int) -> None:
    """Visualize ROC curves for all tickers."""
    plt.figure(figsize=(12, 8))
    for ticker, res in trained.items():
        y_test = res['y_test_actual']
        y_proba = res['y_test_proba']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{ticker} (AUC={roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Test Set (Past={past_window}d, Future={future_window}d)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    avg_auc = np.mean([res['eval_optimized']['auc'] for res in trained.values()])
    plt.text(0.6, 0.3, f'Average AUC: {avg_auc:.3f}', 
            transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()


def visualize_f1_comparison(trained: Dict, tickers: List[str]) -> None:
    """Visualize F1 score comparison: Baseline vs Optimized."""
    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(tickers))
    width = 0.35

    baseline_f1_list = []
    optimized_f1_list = []

    for ticker in tickers:
        res = trained[ticker]
        baseline = res['eval_baseline']
        optimized = res['eval_optimized']
        
        baseline_f1 = baseline['f1_per_class'][1] if len(baseline['f1_per_class']) > 1 else baseline['f1_per_class'][0]
        optimized_f1 = optimized['f1_per_class'][1] if len(optimized['f1_per_class']) > 1 else optimized['f1_per_class'][0]
        
        baseline_f1_list.append(baseline_f1)
        optimized_f1_list.append(optimized_f1)

    bars1 = ax.bar(x_pos - width/2, baseline_f1_list, width, label='Baseline (0.5)', 
                alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x_pos + width/2, optimized_f1_list, width, label='Optimized Threshold', 
                alpha=0.8, color='lightgreen', edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Stock', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test F1 Score (UP class)', fontsize=13, fontweight='bold')
    ax.set_title('Test Set: Baseline vs Optimized Threshold Performance', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tickers, fontsize=11, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()


def visualize_confusion_matrices(trained: Dict, tickers: List[str]) -> None:
    """Visualize confusion matrices for each ticker."""
    for ticker in tickers:
        res = trained[ticker]
        y_test = res['y_test_actual']
        y_proba = res['y_test_proba']
        threshold_used = res.get('threshold_used', res['best_threshold'])
        
        # Predictions at both thresholds
        y_pred_baseline = (y_proba >= 0.5).astype(int)
        y_pred_used = (y_proba >= threshold_used).astype(int)
        
        # Confusion matrices
        cm_baseline = confusion_matrix(y_test, y_pred_baseline)
        cm_used = confusion_matrix(y_test, y_pred_used)
        
        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{ticker} - Confusion Matrix Comparison (Test Set)', 
                    fontsize=14, fontweight='bold')
        
        # Baseline
        im1 = ax1.imshow(cm_baseline, cmap='Blues', alpha=0.7)
        ax1.set_title(f'Baseline (threshold=0.5)\nF1={res["eval_baseline"]["f1_per_class"][1] if len(res["eval_baseline"]["f1_per_class"]) > 1 else res["eval_baseline"]["f1_per_class"][0]:.3f}', 
                    fontsize=12, fontweight='bold')
        for i in range(cm_baseline.shape[0]):
            for j in range(cm_baseline.shape[1]):
                ax1.text(j, i, str(cm_baseline[i, j]), ha='center', va='center',
                        fontsize=20, fontweight='bold', color='black')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Pred DOWN', 'Pred UP'], fontsize=10)
        ax1.set_yticklabels(['Actual DOWN', 'Actual UP'], fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Threshold Used
        used_f1 = res['eval_optimized']['f1_per_class'][1] if len(res['eval_optimized']['f1_per_class']) > 1 else res['eval_optimized']['f1_per_class'][0]
        im2 = ax2.imshow(cm_used, cmap='Greens', alpha=0.7)
        ax2.set_title(f'Threshold Used ({threshold_used:.3f})\nF1={used_f1:.3f}', 
                    fontsize=12, fontweight='bold')
        for i in range(cm_used.shape[0]):
            for j in range(cm_used.shape[1]):
                ax2.text(j, i, str(cm_used[i, j]), ha='center', va='center',
                        fontsize=20, fontweight='bold', color='black')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Pred DOWN', 'Pred UP'], fontsize=10)
        ax2.set_yticklabels(['Actual DOWN', 'Actual UP'], fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        plt.show()
