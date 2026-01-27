"""
Visualization Module for Optimization Benchmarking.

Provides plotting functions for:
- Comparing optimization methods
- Analyzing convergence
- Parameter recovery
- Start point distributions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Pre-declare plotting symbols for static type checkers
plt: Any = None
gridspec: Any = None
Patch: Any = None
sns: Any = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    gridspec = None
    Patch = None
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False


def check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


class BenchmarkVisualizer:
    """
    Visualization tools for benchmark results.
    """
    
    def __init__(self, figsize_base: Tuple[int, int] = (6, 4), dpi: int = 150):
        check_matplotlib()
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Color palette
        if HAS_SEABORN:
            self.colors = sns.color_palette("tab10", 10)
        else:
            try:
                self.colors = list(plt.get_cmap("tab10").colors)
            except Exception:
                self.colors = ["#1f77b4"] * 10
    
    def plot_method_comparison(
        self,
        method_results: Dict[str, Any],  # Dict of MethodResults
        metric: str = 'best_llh',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Bar plot comparing methods by a metric.
        
        Parameters
        ----------
        method_results : dict
            Dictionary mapping method names to MethodResults objects
        metric : str
            Metric to compare: 'best_llh', 'mean_llh', 'mean_fun_calls', etc.
        """
        check_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize_base, dpi=self.dpi)
        
        methods = []
        values = []
        errors = []
        
        for method, results in method_results.items():
            stats = results.get_summary_stats()
            methods.append(results.abbr)
            
            if metric == 'best_llh':
                values.append(stats['best_llh'])
                errors.append(0)
            elif metric == 'mean_llh':
                values.append(stats['mean_llh'])
                errors.append(stats['std_llh'])
            elif metric == 'mean_fun_calls':
                values.append(stats['mean_fun_calls'])
                errors.append(stats['std_fun_calls'])
            else:
                values.append(stats.get(metric, np.nan))
                errors.append(0)
        
        x = np.arange(len(methods))
        colors = [self.colors[i % len(self.colors)] for i in range(len(methods))]
        
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        if any(e > 0 for e in errors):
            ax.errorbar(x, values, yerr=errors, fmt='none', color='black', capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_convergence_distribution(
        self,
        method_results: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Box plot of log-likelihood distributions across methods.
        """
        check_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize_base, dpi=self.dpi)
        
        data = []
        labels = []
        
        for method, results in method_results.items():
            max_llhs = results.get_max_llhs()
            valid_llhs = max_llhs[np.isfinite(max_llhs)]
            if len(valid_llhs) > 0:
                data.append(valid_llhs)
                labels.append(results.abbr)
        
        if HAS_SEABORN and sns is not None:
            import pandas as pd
            df_data = []
            for llhs, label in zip(data, labels):
                for llh in llhs:
                    df_data.append({'Method': label, 'Max Log-Likelihood': llh})
            df = pd.DataFrame(df_data)
            sns.boxplot(data=df, x='Method', y='Max Log-Likelihood', ax=ax, 
                       palette=self.colors[:len(labels)])
        else:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Max Log-Likelihood')
        if title:
            ax.set_title(title)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_efficiency(
        self,
        method_results: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Scatter plot: Function calls vs Best Log-Likelihood.
        
        Efficient methods are in the upper-left (few calls, high likelihood).
        """
        check_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize_base, dpi=self.dpi)
        
        for i, (method, results) in enumerate(method_results.items()):
            stats = results.get_summary_stats()
            
            ax.scatter(
                stats['mean_fun_calls'],
                stats['best_llh'],
                s=100,
                c=[self.colors[i % len(self.colors)]],
                label=results.abbr,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.8
            )
            
            # Error bars
            ax.errorbar(
                stats['mean_fun_calls'],
                stats['mean_llh'],
                xerr=stats['std_fun_calls'],
                yerr=stats['std_llh'],
                fmt='none',
                color=self.colors[i % len(self.colors)],
                alpha=0.5,
                capsize=3
            )
        
        ax.set_xlabel('Mean Function Calls')
        ax.set_ylabel('Best Log-Likelihood')
        ax.set_xscale('log')
        ax.legend(loc='lower right')
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_multistart_landscape(
        self,
        result,  # Result object
        param_idx_x: int = 0,
        param_idx_y: int = 1,
        param_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot optimization endpoints in 2D parameter space.
        
        Useful for visualizing basin structure in multistart optimization.
        """
        check_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize_base, dpi=self.dpi)
        
        if not hasattr(result, 'posterior_samples'):
            raise ValueError("Result does not have posterior_samples")
        
        samples = result.posterior_samples
        llhs = result.posterior_llhs
        
        # Color by likelihood
        sc = ax.scatter(
            samples[:, param_idx_x],
            samples[:, param_idx_y],
            c=llhs,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Mark best point
        best_idx = np.argmax(llhs)
        ax.scatter(
            samples[best_idx, param_idx_x],
            samples[best_idx, param_idx_y],
            c='red',
            s=200,
            marker='*',
            edgecolors='black',
            linewidth=1,
            label='Best'
        )
        
        plt.colorbar(sc, ax=ax, label='Log-Likelihood')
        
        if param_names:
            ax.set_xlabel(param_names[param_idx_x])
            ax.set_ylabel(param_names[param_idx_y])
        else:
            ax.set_xlabel(f'Parameter {param_idx_x}')
            ax.set_ylabel(f'Parameter {param_idx_y}')
        
        ax.legend()
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_parameter_recovery(
        self,
        result,  # Result object
        true_params: np.ndarray,
        param_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Compare estimated parameters to true values.
        """
        check_matplotlib()
        fig, ax = plt.subplots(figsize=(max(8, len(true_params) * 0.8), 5), dpi=self.dpi)
        
        estimated = result.get_best_parameters()
        if estimated is None:
            raise ValueError("Could not extract best parameters from result")
        
        n_params = len(true_params)
        x = np.arange(n_params)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, true_params, width, label='True', 
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, estimated, width, label='Estimated',
                      color='coral', alpha=0.8, edgecolor='black')
        
        if param_names:
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([f'P{i}' for i in range(n_params)])
        
        ax.set_ylabel('Parameter Value')
        ax.legend()
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_start_vs_final(
        self,
        result,  # Result object with multistart info
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Scatter plot: Starting point quality vs final objective value.
        
        For multistart methods only.
        """
        check_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize_base, dpi=self.dpi)
        
        if not hasattr(result, 'algo_specific_info'):
            raise ValueError("Result does not have algo_specific_info")
        
        info = result.algo_specific_info
        if 'all_fvals' not in info:
            raise ValueError("Result does not have start point information")
        
        # Get final values
        final_fvals = np.array(info['all_fvals'])
        
        # Filter valid
        valid_mask = final_fvals < 1e9
        final_valid = final_fvals[valid_mask]
        
        if len(final_valid) == 0:
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center', 
                   transform=ax.transAxes)
            return fig, ax
        
        # Success coloring
        if 'all_success' in info:
            success = np.array(info['all_success'])[valid_mask]
            colors = ['green' if s else 'red' for s in success]
        else:
            colors = 'steelblue'
        
        # Plot sorted final values
        sorted_idx = np.argsort(final_valid)
        ax.scatter(
            range(len(final_valid)),
            final_valid[sorted_idx],
            c=[colors[i] for i in sorted_idx] if isinstance(colors, list) else colors,
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Optimization Run (sorted by final value)')
        ax.set_ylabel('Final Objective Value')
        ax.set_yscale('log')
        
        # Add legend for success/failure
        if 'all_success' in info:
            if Patch is not None:
                legend_elements = [
                    Patch(facecolor='green', edgecolor='black', label='Success'),
                    Patch(facecolor='red', edgecolor='black', label='Failure')
                ]
                ax.legend(handles=legend_elements, loc='upper left')
            else:
                ax.legend(loc='upper left')
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_comprehensive_comparison(
        self,
        method_results: Dict[str, Any],
        problem_name: str,
        save_path: Optional[str] = None
    ):
        """
        Create a comprehensive multi-panel comparison figure.
        """
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Best likelihood comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric_bars(ax1, method_results, 'best_llh', 'Best Log-Likelihood')
        
        # Panel 2: Function calls
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric_bars(ax2, method_results, 'mean_fun_calls', 'Mean Function Calls')
        ax2.set_yscale('log')
        
        # Panel 3: Likelihood distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_likelihood_box(ax3, method_results)
        
        # Panel 4: Efficiency scatter
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_efficiency_scatter(ax4, method_results)
        
        fig.suptitle(f'Benchmark Comparison: {problem_name}', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_metric_bars(self, ax, method_results, metric, ylabel):
        """Helper for bar plots."""
        methods = []
        values = []
        
        for method, results in method_results.items():
            stats = results.get_summary_stats()
            methods.append(results.abbr)
            values.append(stats.get(metric, np.nan))
        
        x = np.arange(len(methods))
        colors = [self.colors[i % len(self.colors)] for i in range(len(methods))]
        ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
    
    def _plot_likelihood_box(self, ax, method_results):
        """Helper for boxplot."""
        data = []
        labels = []
        
        for method, results in method_results.items():
            max_llhs = results.get_max_llhs()
            valid_llhs = max_llhs[np.isfinite(max_llhs)]
            if len(valid_llhs) > 0:
                data.append(valid_llhs)
                labels.append(results.abbr)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Max Log-Likelihood')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    def _plot_efficiency_scatter(self, ax, method_results):
        """Helper for efficiency scatter."""
        for i, (method, results) in enumerate(method_results.items()):
            stats = results.get_summary_stats()
            ax.scatter(
                stats['mean_fun_calls'],
                stats['best_llh'],
                s=100,
                c=[self.colors[i % len(self.colors)]],
                label=results.abbr,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.8
            )
        
        ax.set_xlabel('Mean Function Calls')
        ax.set_ylabel('Best Log-Likelihood')
        ax.set_xscale('log')
        ax.legend(loc='lower right', fontsize=8)


def quick_plot(method_results: Dict[str, Any], problem_name: str = "Problem"):
    """
    Quick visualization of benchmark results.
    
    Parameters
    ----------
    method_results : dict
        Dictionary mapping method names to MethodResults objects
    problem_name : str
        Name of the problem for the title
    """
    check_matplotlib()
    viz = BenchmarkVisualizer()
    return viz.plot_comprehensive_comparison(method_results, problem_name)