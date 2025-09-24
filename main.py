import os
# Disable GPU for TensorFlow and related libraries
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=""'  # Disable XLA GPU
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force JAX to use CPU

# Suppress CUDA warnings
import warnings
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*cuFFT.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from adapt.instance_based import KLIEP, KMM, TrAdaBoost
from src.cross_adaptation.core.cross_adaptation import Adapter
from src.visualization import AdaptationVisualizer, load_original_and_adapted_data
from lazypredict.Supervised import LazyClassifier
import wandb
import hydra
import random
from statistics import mean
from omegaconf import DictConfig

@hydra.main(
    version_base="1.1", config_path="experiments/config", config_name="config"
)
def main(cfg: DictConfig):
    """Main function for cross-adaptation experiments"""
    train_data = {
        k: pd.concat([
            pd.read_csv(os.path.join(cfg.root_path, 'train', df), index_col=0),
            pd.read_csv(os.path.join(cfg.root_path, 'test', df), index_col=0)
        ], ignore_index=True)
        for k, df in cfg.train_data.items() 
    }
    test_data = {
        k: pd.read_csv(os.path.join(cfg.root_path, 'test', df), index_col=0) for k, df in cfg.test_data.items()
    }
    # Create estimator and adapt_model
    if cfg.classifier._target_ == "sklearn.neighbors.KNeighborsClassifier":
        estimator = hydra.utils.instantiate(cfg.classifier)
    else:
        estimator = hydra.utils.instantiate(cfg.classifier, random_state=cfg.random_state)
    if cfg.adapt_model._target_ == "adapt.instance_based.WANN":
        adapt_model = hydra.utils.instantiate(
            cfg.adapt_model, random_state=cfg.random_state
        )
    else:
        adapt_model = hydra.utils.instantiate(
            cfg.adapt_model, estimator=estimator, random_state=cfg.random_state
        )

    # # Configure XGBoost to use CPU
    # estimator = RandomForestClassifier()
    # # estimator = XGBClassifier(
    # #     tree_method='hist',  # Use histogram-based algorithm for CPU
    # #     device='cpu'  # Explicitly set to use CPU
    # # )
    # adapt_model = adapt_model = KLIEP(
    #     kernel='rbf',
    #     max_centers=100,
    #     lr=0.01,
    #     max_iter=2000
    # )
    # Convert config to dict and filter out non-serializable keys
    config_dict = {}
    for key, value in cfg.items():
        try:
            # Test if the value is JSON serializable
            import json
            json.dumps({key: value})
            config_dict[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable values
            continue
    
    run = wandb.init(
                entity="thelion-ai",
                project="cross-adaptation-5",
                config=config_dict,
                name=f"{cfg.classifier._target_}_{cfg.adapt_model._target_}_{random.randint(0, 100000)}",
            )
    adapter = Adapter(
        train_data=train_data, estimator=estimator, adapt_model=adapt_model
    )
    
    # Adapt the data and save it to a directory
    save_dir = os.path.join("outputs", "adapted_data")
    adapted_data = adapter.adapt(save_dir=save_dir)
    
    # Create visualizations comparing original vs adapted data
    if cfg.get('create_visualizations', True):
        print("Creating adaptation visualizations...")
        viz = AdaptationVisualizer()
        
        # Load original test data for comparison
        original_test_data = {
            k: pd.read_csv(os.path.join(cfg.root_path, 'test', df), index_col=0) 
            for k, df in cfg.test_data.items()
        }
        
        # Create visualization directory
        viz_dir = os.path.join("outputs", "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get available blood parameters
        sample_df = list(test_data.values())[0]
        blood_params = [col for col in sample_df.columns 
                       if col != 'target' and pd.api.types.is_numeric_dtype(sample_df[col])][:6]  # Limit to 6 for readability
        
        # Distribution comparison
        viz.plot_distribution_comparison(
            original_test_data, adapted_data,
            countries=list(cfg.test_data.keys()),
            blood_params=blood_params,
            save_path=os.path.join(viz_dir, "distribution_comparison.png")
        )
        
        # Scatter plot comparison (using first two numeric parameters)
        if len(blood_params) >= 2:
            viz.plot_scatter_comparison(
                original_test_data, adapted_data,
                x_param=blood_params[0], y_param=blood_params[1],
                countries=list(cfg.test_data.keys()),
                save_path=os.path.join(viz_dir, "scatter_comparison.png")
            )
        
        # Statistical comparison
        stats_df = viz.plot_statistics_comparison(
            original_test_data, adapted_data,
            countries=list(cfg.test_data.keys()),
            blood_params=blood_params,
            save_path=os.path.join(viz_dir, "statistics_comparison.png")
        )
        
        # Save statistics to CSV
        stats_df.to_csv(os.path.join(viz_dir, "adaptation_statistics.csv"), index=False)
        print(f"Visualizations saved to: {viz_dir}")
    
    # Log adapted data info if using wandb

    # Prepare metrics
    metrics = [
        hydra.utils.instantiate(metric, _partial_=True, zero_division=0.0) \
        if metric['_target_'] not in ['sklearn.metrics.accuracy_score', 'sklearn.metrics.roc_auc_score'] \
        else hydra.utils.instantiate(metric, _partial_=True) \
        for metric in cfg.metrics
    ]
    
    # Train on adapted data and evaluate
    results = adapter.train_on_adapted_data(
        test_data=test_data, 
        metrics=metrics,
        use_weights=True,  # Use instance weights from adaptation
        save_model=True    # Save the trained model
    )
    # Log adapted results
    # print("Adapted model results:", results)
    wandb.log(results)
    
    # Calculate baseline results
    baseline_results = adapter.calc_baseline(test_data, metrics=metrics)
    # print("Baseline model results:", baseline_results)
    wandb.log(baseline_results)
    
    # Compare adapted vs baseline
    compare_results = adapter.compare(results, baseline_results, metrics, test_data)
    # print("Comparison (adapted - baseline):", compare_results)
    wandb.log(compare_results)
    # Calculate the score for optuna
    score = dict(
        filter(lambda item: cfg.optimized_metric in item[0], results.items())
    ).values()
    score = mean(list(score))
    print(f"Mean {cfg.optimized_metric} score: {score}")
    wandb.log({f"{cfg.optimized_metric}_mean_score": score})
    
    # # Print summary
    # print("\n" + "="*60)
    # print("EXPERIMENT SUMMARY")
    # print("="*60)
    # print(f"Adaptation method: {adapt_model.__class__.__name__}")
    # print(f"Estimator: {estimator.__class__.__name__}")
    # print(f"Adapted data saved to: {save_dir}")
    # print(f"Number of datasets adapted: {len(adapted_data)}")
    # print(f"Mean optimization metric ({cfg.optimized_metric}): {score:.4f}")
    # print("="*60)
    
    wandb.finish()
    # print("\nExperiment completed successfully!")
    return score
    
if __name__ == "__main__":
    main()
