import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os

class AdaptationVisualizer:
    """Visualize data before and after cross adaptation"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_distribution_comparison(self, 
                                   original_data: Dict[str, pd.DataFrame], 
                                   adapted_data: Dict[str, pd.DataFrame],
                                   countries: Optional[List[str]] = None,
                                   blood_params: Optional[List[str]] = None,
                                   save_path: Optional[str] = None):
        """
        Compare distributions of blood parameters before and after adaptation
        
        Args:
            original_data: Dictionary of original datasets {country: dataframe}
            adapted_data: Dictionary of adapted datasets {country: dataframe}
            countries: List of countries to visualize (None for all)
            blood_params: List of blood parameters to visualize (None for all)
            save_path: Path to save the plot
        """
        
        # Filter countries
        if countries is None:
            countries = list(original_data.keys())
        else:
            countries = [c for c in countries if c in original_data.keys()]
        
        # Get blood parameters (exclude target and non-numeric columns)
        if blood_params is None:
            sample_df = list(original_data.values())[0]
            blood_params = [col for col in sample_df.columns 
                           if col != 'target' and pd.api.types.is_numeric_dtype(sample_df[col])]
        
        n_params = len(blood_params)
        n_countries = len(countries)
        
        # Create subplots
        fig, axes = plt.subplots(n_params, n_countries, figsize=(5*n_countries, 4*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
        if n_countries == 1:
            axes = axes.reshape(-1, 1)
        
        for i, param in enumerate(blood_params):
            for j, country in enumerate(countries):
                ax = axes[i, j] if n_params > 1 or n_countries > 1 else axes
                
                # Get data for this country and parameter
                orig_values = original_data[country][param].dropna()
                adapted_values = adapted_data[country][param].dropna()
                
                # Plot distributions
                ax.hist(orig_values, alpha=0.6, label='Original', density=True, bins=30)
                ax.hist(adapted_values, alpha=0.6, label='Adapted', density=True, bins=30)
                
                ax.set_title(f'{country} - {param}')
                ax.set_xlabel(param)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter_comparison(self,
                              original_data: Dict[str, pd.DataFrame],
                              adapted_data: Dict[str, pd.DataFrame],
                              x_param: str,
                              y_param: str,
                              countries: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
        """
        Create scatter plots comparing two blood parameters before and after adaptation
        
        Args:
            original_data: Dictionary of original datasets
            adapted_data: Dictionary of adapted datasets
            x_param: Blood parameter for x-axis
            y_param: Blood parameter for y-axis
            countries: List of countries to visualize
            save_path: Path to save the plot
        """
        
        if countries is None:
            countries = list(original_data.keys())
        
        n_countries = len(countries)
        fig, axes = plt.subplots(2, n_countries, figsize=(5*n_countries, 10))
        
        if n_countries == 1:
            axes = axes.reshape(-1, 1)
        
        for j, country in enumerate(countries):
            # Original data
            orig_df = original_data[country]
            ax_orig = axes[0, j] if n_countries > 1 else axes[0]
            
            colors = ['red' if target == 1 else 'blue' for target in orig_df['target']]
            ax_orig.scatter(orig_df[x_param], orig_df[y_param], 
                           c=colors, alpha=0.6, s=20)
            ax_orig.set_title(f'{country} - Original')
            ax_orig.set_xlabel(x_param)
            ax_orig.set_ylabel(y_param)
            ax_orig.grid(True, alpha=0.3)
            
            # Adapted data
            adapted_df = adapted_data[country]
            ax_adapted = axes[1, j] if n_countries > 1 else axes[1]
            
            colors = ['red' if target == 1 else 'blue' for target in adapted_df['target']]
            ax_adapted.scatter(adapted_df[x_param], adapted_df[y_param], 
                              c=colors, alpha=0.6, s=20)
            ax_adapted.set_title(f'{country} - Adapted')
            ax_adapted.set_xlabel(x_param)
            ax_adapted.set_ylabel(y_param)
            ax_adapted.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='COVID+'),
                          Patch(facecolor='blue', label='COVID-')]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_statistics_comparison(self,
                                 original_data: Dict[str, pd.DataFrame],
                                 adapted_data: Dict[str, pd.DataFrame],
                                 countries: Optional[List[str]] = None,
                                 blood_params: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """
        Compare statistical measures (mean, std, median) before and after adaptation
        
        Args:
            original_data: Dictionary of original datasets
            adapted_data: Dictionary of adapted datasets
            countries: List of countries to visualize
            blood_params: List of blood parameters to analyze
            save_path: Path to save the plot
        """
        
        if countries is None:
            countries = list(original_data.keys())
        
        if blood_params is None:
            sample_df = list(original_data.values())[0]
            blood_params = [col for col in sample_df.columns 
                           if col != 'target' and pd.api.types.is_numeric_dtype(sample_df[col])]
        
        # Collect statistics
        stats_data = []
        
        for country in countries:
            orig_df = original_data[country]
            adapted_df = adapted_data[country]
            
            for param in blood_params:
                # Original stats
                stats_data.append({
                    'Country': country,
                    'Parameter': param,
                    'Type': 'Original',
                    'Mean': orig_df[param].mean(),
                    'Std': orig_df[param].std(),
                    'Median': orig_df[param].median()
                })
                
                # Adapted stats
                stats_data.append({
                    'Country': country,
                    'Parameter': param,
                    'Type': 'Adapted',
                    'Mean': adapted_df[param].mean(),
                    'Std': adapted_df[param].std(),
                    'Median': adapted_df[param].median()
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Mean comparison
        sns.barplot(data=stats_df, x='Parameter', y='Mean', hue='Type', ax=axes[0])
        axes[0].set_title('Mean Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Std comparison
        sns.barplot(data=stats_df, x='Parameter', y='Std', hue='Type', ax=axes[1])
        axes[1].set_title('Standard Deviation Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Median comparison
        sns.barplot(data=stats_df, x='Parameter', y='Median', hue='Type', ax=axes[2])
        axes[2].set_title('Median Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def plot_correlation_heatmap(self,
                               original_data: Dict[str, pd.DataFrame],
                               adapted_data: Dict[str, pd.DataFrame],
                               countries: Optional[List[str]] = None,
                               blood_params: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
        """
        Compare correlation matrices before and after adaptation
        
        Args:
            original_data: Dictionary of original datasets
            adapted_data: Dictionary of adapted datasets
            countries: List of countries to visualize
            blood_params: List of blood parameters to analyze
            save_path: Path to save the plot
        """
        
        if countries is None:
            countries = list(original_data.keys())
        
        if blood_params is None:
            sample_df = list(original_data.values())[0]
            blood_params = [col for col in sample_df.columns 
                           if col != 'target' and pd.api.types.is_numeric_dtype(sample_df[col])]
        
        n_countries = len(countries)
        fig, axes = plt.subplots(2, n_countries, figsize=(8*n_countries, 16))
        
        if n_countries == 1:
            axes = axes.reshape(-1, 1)
        
        for j, country in enumerate(countries):
            # Original correlation
            orig_corr = original_data[country][blood_params].corr()
            ax_orig = axes[0, j] if n_countries > 1 else axes[0]
            sns.heatmap(orig_corr, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', ax=ax_orig)
            ax_orig.set_title(f'{country} - Original Correlations')
            
            # Adapted correlation
            adapted_corr = adapted_data[country][blood_params].corr()
            ax_adapted = axes[1, j] if n_countries > 1 else axes[1]
            sns.heatmap(adapted_corr, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', ax=ax_adapted)
            ax_adapted.set_title(f'{country} - Adapted Correlations')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_original_and_adapted_data(original_path: str, adapted_path: str) -> tuple:
    """
    Load original and adapted datasets from directories
    
    Args:
        original_path: Path to directory containing original CSV files
        adapted_path: Path to directory containing adapted CSV files
    
    Returns:
        Tuple of (original_data_dict, adapted_data_dict)
    """
    original_data = {}
    adapted_data = {}
    
    # Load original data
    for file in os.listdir(original_path):
        if file.endswith('.csv'):
            country = file.replace('.csv', '')
            original_data[country] = pd.read_csv(os.path.join(original_path, file))
    
    # Load adapted data
    for file in os.listdir(adapted_path):
        if file.endswith('.csv'):
            country = file.replace('.csv', '')
            adapted_data[country] = pd.read_csv(os.path.join(adapted_path, file))
    
    return original_data, adapted_data


# Example usage
if __name__ == "__main__":
    # Load data
    original_path = "experiments/data/tabular/processed/test"
    adapted_path = "outputs/adapted_data"
    
    original_data, adapted_data = load_original_and_adapted_data(original_path, adapted_path)
    
    # Create visualizer
    viz = AdaptationVisualizer()
    
    # Example visualizations
    viz.plot_distribution_comparison(
        original_data, adapted_data,
        countries=['spain', 'ethiopia'],
        blood_params=['HGB', 'WBC', 'PLT'],
        save_path='adaptation_distributions.png'
    )
    
    viz.plot_scatter_comparison(
        original_data, adapted_data,
        x_param='HGB', y_param='WBC',
        countries=['spain', 'ethiopia'],
        save_path='adaptation_scatter.png'
    )