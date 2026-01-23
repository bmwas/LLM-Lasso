"""
Bootstrap Confidence Interval Utilities

Provides functions for computing bootstrap confidence intervals for classification metrics
using the mlxtend library.

Supports:
- Standard bootstrap (simple resampling with replacement)
- .632 bootstrap (reduces optimistic bias)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
import logging

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

# mlxtend bootstrap
try:
    from mlxtend.evaluate import bootstrap
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _create_metric_func(metric_name: str, y_prob: Optional[np.ndarray] = None) -> Callable:
    """
    Create a metric function that takes (y_true, y_pred) and returns a scalar.
    
    For probability-based metrics (auroc, avg_precision), we use the stored y_prob.
    """
    if metric_name == 'accuracy':
        return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    elif metric_name == 'balanced_accuracy':
        return lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred)
    elif metric_name == 'sensitivity':
        return lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
    elif metric_name == 'specificity':
        return lambda y_true, y_pred: specificity_score(y_true, y_pred)
    elif metric_name == 'precision':
        return lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0)
    elif metric_name == 'f1':
        return lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0)
    elif metric_name == 'mcc':
        return lambda y_true, y_pred: matthews_corrcoef(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def _bootstrap_metric_standard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_rounds: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric using standard bootstrap.
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    # Compute point estimate
    point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_rounds):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except Exception:
            continue
    
    if len(bootstrap_scores) < 10:
        # Not enough valid bootstrap samples
        return point_estimate, np.nan, np.nan
    
    # Compute confidence intervals using percentile method
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def _bootstrap_proba_metric_standard(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func: Callable,
    n_rounds: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for probability-based metrics.
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    # Compute point estimate
    try:
        point_estimate = metric_func(y_true, y_prob)
    except Exception:
        return np.nan, np.nan, np.nan
    
    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_rounds):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            score = metric_func(y_true_boot, y_prob_boot)
            bootstrap_scores.append(score)
        except Exception:
            continue
    
    if len(bootstrap_scores) < 10:
        return point_estimate, np.nan, np.nan
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def _bootstrap_metric_632(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_rounds: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute .632 bootstrap confidence interval for a metric.
    
    The .632 bootstrap adjusts for optimistic bias by weighting:
    err_632 = 0.368 * err_train + 0.632 * err_oob
    
    For our case with fixed predictions, we use a modified approach that
    weights in-bag and out-of-bag samples.
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    # Compute point estimate on full data
    point_estimate = metric_func(y_true, y_pred)
    
    bootstrap_scores = []
    for _ in range(n_rounds):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Out-of-bag indices
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        oob_indices = np.where(oob_mask)[0]
        
        if len(oob_indices) < 2:
            continue
        
        # In-bag score
        y_true_inbag = y_true[indices]
        y_pred_inbag = y_pred[indices]
        
        # Out-of-bag score
        y_true_oob = y_true[oob_indices]
        y_pred_oob = y_pred[oob_indices]
        
        # Skip if only one class
        if len(np.unique(y_true_inbag)) < 2 or len(np.unique(y_true_oob)) < 2:
            continue
        
        try:
            score_inbag = metric_func(y_true_inbag, y_pred_inbag)
            score_oob = metric_func(y_true_oob, y_pred_oob)
            
            # .632 weighting
            score_632 = 0.368 * score_inbag + 0.632 * score_oob
            bootstrap_scores.append(score_632)
        except Exception:
            continue
    
    if len(bootstrap_scores) < 10:
        return point_estimate, np.nan, np.nan
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    method: str = "standard",
    n_rounds: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for all classification metrics.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        method: Bootstrap method - "standard" or "632" (.632 bootstrap)
        n_rounds: Number of bootstrap iterations
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        logger: Optional logger for status messages
    
    Returns:
        Dictionary with structure:
        {
            'metric_name': {
                'point_estimate': float,
                'ci_lower': float,
                'ci_upper': float,
                'ci_level': float
            },
            ...
        }
    """
    if logger:
        logger.info(f"Computing bootstrap confidence intervals ({method} method, {n_rounds} rounds, {ci_level*100:.0f}% CI)")
    
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    
    # Select bootstrap function based on method
    if method == "632":
        bootstrap_func = _bootstrap_metric_632
    else:
        bootstrap_func = _bootstrap_metric_standard
    
    results = {}
    
    # Classification metrics (based on y_pred)
    classification_metrics = [
        'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity',
        'precision', 'f1', 'mcc'
    ]
    
    for metric_name in classification_metrics:
        metric_func = _create_metric_func(metric_name)
        point_est, ci_lower, ci_upper = bootstrap_func(
            y_true, y_pred, metric_func,
            n_rounds=n_rounds, ci_level=ci_level, random_seed=random_seed
        )
        
        results[metric_name] = {
            'point_estimate': float(point_est),
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
            'ci_level': ci_level
        }
        
        if logger:
            if results[metric_name]['ci_lower'] is not None:
                logger.debug(f"  {metric_name}: {point_est:.4f} ({ci_level*100:.0f}% CI: {ci_lower:.4f} - {ci_upper:.4f})")
            else:
                logger.debug(f"  {metric_name}: {point_est:.4f} (CI not computed)")
    
    # Probability-based metrics (based on y_prob)
    prob_metrics = {
        'auroc': lambda y, p: roc_auc_score(y, p),
        'average_precision': lambda y, p: average_precision_score(y, p)
    }
    
    for metric_name, metric_func in prob_metrics.items():
        point_est, ci_lower, ci_upper = _bootstrap_proba_metric_standard(
            y_true, y_prob, metric_func,
            n_rounds=n_rounds, ci_level=ci_level, random_seed=random_seed
        )
        
        results[metric_name] = {
            'point_estimate': float(point_est) if not np.isnan(point_est) else None,
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
            'ci_level': ci_level
        }
        
        if logger and results[metric_name]['point_estimate'] is not None:
            if results[metric_name]['ci_lower'] is not None:
                logger.debug(f"  {metric_name}: {point_est:.4f} ({ci_level*100:.0f}% CI: {ci_lower:.4f} - {ci_upper:.4f})")
            else:
                logger.debug(f"  {metric_name}: {point_est:.4f} (CI not computed)")
    
    if logger:
        logger.info(f"Bootstrap confidence intervals computed for {len(results)} metrics")
    
    return results


def format_metric_with_ci(
    metric_name: str,
    ci_result: Dict[str, float],
    decimal_places: int = 4
) -> str:
    """
    Format a metric value with its confidence interval for display.
    
    Args:
        metric_name: Name of the metric
        ci_result: Dict with 'point_estimate', 'ci_lower', 'ci_upper', 'ci_level'
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string like "Accuracy: 0.8523 (95% CI: 0.7821 - 0.9104)"
    """
    point_est = ci_result.get('point_estimate')
    ci_lower = ci_result.get('ci_lower')
    ci_upper = ci_result.get('ci_upper')
    ci_level = ci_result.get('ci_level', 0.95)
    
    if point_est is None:
        return f"{metric_name}: N/A"
    
    fmt = f".{decimal_places}f"
    
    if ci_lower is not None and ci_upper is not None:
        ci_pct = int(ci_level * 100)
        return f"{metric_name}: {point_est:{fmt}} ({ci_pct}% CI: {ci_lower:{fmt}} - {ci_upper:{fmt}})"
    else:
        return f"{metric_name}: {point_est:{fmt}}"


def plot_confidence_intervals(
    ci_results: Dict[str, Dict[str, float]],
    save_dir: str,
    ci_level: float = 0.95,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Generate and save confidence interval visualization plots.
    
    Args:
        ci_results: Dictionary of CI results from compute_bootstrap_ci()
        save_dir: Directory to save plots (confidence_intervals subfolder)
        ci_level: Confidence level for labeling
        logger: Optional logger
    
    Returns:
        List of saved file paths
    """
    import matplotlib.pyplot as plt
    import os
    
    saved_files = []
    ci_pct = int(ci_level * 100)
    
    # Define metric display names and order
    metric_display = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Accuracy',
        'sensitivity': 'Sensitivity (Recall)',
        'specificity': 'Specificity',
        'precision': 'Precision',
        'f1': 'F1 Score',
        'auroc': 'AUROC',
        'average_precision': 'Avg Precision',
        'mcc': 'MCC'
    }
    
    # Filter to metrics that have valid CIs
    valid_metrics = []
    for metric in metric_display.keys():
        if metric in ci_results:
            ci = ci_results[metric]
            if ci.get('point_estimate') is not None and ci.get('ci_lower') is not None:
                valid_metrics.append(metric)
    
    if not valid_metrics:
        if logger:
            logger.warning("No valid metrics with CIs to plot")
        return saved_files
    
    # Prepare data for plotting
    names = [metric_display[m] for m in valid_metrics]
    point_estimates = [ci_results[m]['point_estimate'] for m in valid_metrics]
    ci_lowers = [ci_results[m]['ci_lower'] for m in valid_metrics]
    ci_uppers = [ci_results[m]['ci_upper'] for m in valid_metrics]
    
    # Calculate error bars (distance from point estimate)
    errors_lower = [pe - cl for pe, cl in zip(point_estimates, ci_lowers)]
    errors_upper = [cu - pe for pe, cu in zip(point_estimates, ci_uppers)]
    
    # ==================== PLOT 1: Horizontal Bar Chart with Error Bars ====================
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(names))
    
    # Create horizontal bar chart
    bars = ax1.barh(y_pos, point_estimates, xerr=[errors_lower, errors_upper],
                    color='steelblue', alpha=0.8, capsize=5, ecolor='darkblue',
                    error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels
    for i, (pe, cl, cu) in enumerate(zip(point_estimates, ci_lowers, ci_uppers)):
        ax1.text(pe + 0.02, i, f'{pe:.3f}', va='center', fontsize=10, fontweight='bold')
        ax1.text(max(cu + 0.01, 0.85), i, f'[{cl:.3f}, {cu:.3f}]', 
                va='center', fontsize=9, color='gray')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=11)
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_xlim(0, 1.15)
    ax1.set_title(f'Classification Metrics with {ci_pct}% Confidence Intervals', 
                  fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend(loc='lower right')
    
    plt.tight_layout()
    file1 = os.path.join(save_dir, 'metrics_with_ci_bars.png')
    fig1.savefig(file1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    saved_files.append(file1)
    
    # ==================== PLOT 2: Forest Plot Style ====================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(names))
    
    # Plot confidence intervals as horizontal lines
    for i, (pe, cl, cu) in enumerate(zip(point_estimates, ci_lowers, ci_uppers)):
        ax2.plot([cl, cu], [i, i], 'b-', linewidth=2, alpha=0.7)
        ax2.plot([cl, cl], [i-0.1, i+0.1], 'b-', linewidth=2)  # Left cap
        ax2.plot([cu, cu], [i-0.1, i+0.1], 'b-', linewidth=2)  # Right cap
    
    # Plot point estimates as diamonds
    ax2.scatter(point_estimates, y_pos, marker='D', s=100, c='darkblue', 
                zorder=5, label='Point Estimate')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=11)
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_xlim(0, 1.05)
    ax2.set_title(f'Forest Plot: {ci_pct}% Confidence Intervals', 
                  fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(loc='lower right')
    
    # Add annotation box with CI width info
    ci_widths = [cu - cl for cl, cu in zip(ci_lowers, ci_uppers)]
    avg_width = np.mean(ci_widths)
    textstr = f'Mean CI width: {avg_width:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    file2 = os.path.join(save_dir, 'forest_plot_ci.png')
    fig2.savefig(file2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    saved_files.append(file2)
    
    # ==================== PLOT 3: CI Width Comparison ====================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    ci_widths = [cu - cl for cl, cu in zip(ci_lowers, ci_uppers)]
    
    bars3 = ax3.bar(names, ci_widths, color='coral', alpha=0.8, edgecolor='darkred')
    
    # Add value labels on bars
    for bar, width in zip(bars3, ci_widths):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{width:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_ylabel('CI Width', fontsize=12)
    ax3.set_xlabel('Metric', fontsize=12)
    ax3.set_title(f'{ci_pct}% Confidence Interval Widths\n(Narrower = More Precise)', 
                  fontsize=14, fontweight='bold')
    ax3.set_ylim(0, max(ci_widths) * 1.2)
    ax3.axhline(y=np.mean(ci_widths), color='blue', linestyle='--', 
                label=f'Mean width: {np.mean(ci_widths):.3f}')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    file3 = os.path.join(save_dir, 'ci_width_comparison.png')
    fig3.savefig(file3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    saved_files.append(file3)
    
    # ==================== PLOT 4: Summary Dashboard ====================
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Key metrics with CIs (subset)
    key_metrics = ['accuracy', 'balanced_accuracy', 'auroc', 'f1']
    key_names = [metric_display.get(m, m) for m in key_metrics if m in ci_results]
    key_pe = [ci_results[m]['point_estimate'] for m in key_metrics if m in ci_results and ci_results[m].get('point_estimate')]
    key_cl = [ci_results[m]['ci_lower'] for m in key_metrics if m in ci_results and ci_results[m].get('ci_lower')]
    key_cu = [ci_results[m]['ci_upper'] for m in key_metrics if m in ci_results and ci_results[m].get('ci_upper')]
    
    if key_pe and key_cl and key_cu:
        key_err_l = [pe - cl for pe, cl in zip(key_pe, key_cl)]
        key_err_u = [cu - pe for pe, cu in zip(key_pe, key_cu)]
        
        axes[0, 0].barh(key_names, key_pe, xerr=[key_err_l, key_err_u],
                        color='teal', alpha=0.8, capsize=4)
        axes[0, 0].set_xlim(0, 1.1)
        axes[0, 0].set_title('Key Metrics', fontsize=12, fontweight='bold')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Top-right: Sensitivity vs Specificity
    if 'sensitivity' in ci_results and 'specificity' in ci_results:
        sens = ci_results['sensitivity']
        spec = ci_results['specificity']
        
        metrics_ss = ['Sensitivity', 'Specificity']
        pe_ss = [sens['point_estimate'], spec['point_estimate']]
        cl_ss = [sens['ci_lower'], spec['ci_lower']]
        cu_ss = [sens['ci_upper'], spec['ci_upper']]
        
        if all(v is not None for v in pe_ss + cl_ss + cu_ss):
            x_pos = [0, 1]
            axes[0, 1].bar(x_pos, pe_ss, 
                          yerr=[[pe-cl for pe, cl in zip(pe_ss, cl_ss)], 
                                [cu-pe for pe, cu in zip(pe_ss, cu_ss)]],
                          color=['darkorange', 'purple'], alpha=0.8, capsize=5, width=0.5)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(metrics_ss)
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].set_title('Sensitivity vs Specificity', fontsize=12, fontweight='bold')
            axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Bottom-left: All metrics point estimates
    axes[1, 0].barh(names, point_estimates, color='steelblue', alpha=0.8)
    axes[1, 0].set_xlim(0, 1.1)
    axes[1, 0].set_title('All Metrics (Point Estimates)', fontsize=12, fontweight='bold')
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Bottom-right: CI widths
    axes[1, 1].bar(range(len(names)), ci_widths, color='coral', alpha=0.8)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels([n[:10] + '...' if len(n) > 10 else n for n in names], 
                               rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_title('CI Widths (Precision)', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=np.mean(ci_widths), color='blue', linestyle='--', alpha=0.7)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    fig4.suptitle(f'Bootstrap Confidence Intervals Summary ({ci_pct}% CI)', 
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    file4 = os.path.join(save_dir, 'ci_summary_dashboard.png')
    fig4.savefig(file4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    saved_files.append(file4)
    
    if logger:
        logger.info(f"  Saved {len(saved_files)} CI visualization plots to {save_dir}")
    
    return saved_files
