"""Cross-adaptation experiment runner with domain adaptation methods."""

import os
import json
import random
import warnings
from statistics import mean
from typing import Dict, List, Callable

import pandas as pd
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig

from src.cross_adaptation.core.cross_adaptation import Adapter
from src.visualization import AdaptationVisualizer


def _configure_environment():
    """Configure environment variables to disable GPU usage and suppress warnings."""
    # Disable GPU for TensorFlow and related libraries
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '-1',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'XLA_FLAGS': '--xla_gpu_cuda_data_dir=""',
        'JAX_PLATFORMS': 'cpu'
    })

    # Suppress CUDA-related warnings
    cuda_warnings = ['.*CUDA.*', '.*cuFFT.*', '.*cuDNN.*', '.*cuBLAS.*']
    for pattern in cuda_warnings:
        warnings.filterwarnings('ignore', message=pattern)


# Configure environment at module import
_configure_environment()

def load_datasets(cfg: DictConfig) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load training and test datasets from configuration."""
    train_data = {}
    for key, filename in cfg.train_data.items():
        train_df = pd.read_csv(os.path.join(cfg.root_path, 'train', filename), index_col=0)
        test_df = pd.read_csv(os.path.join(cfg.root_path, 'test', filename), index_col=0)
        train_data[key] = pd.concat([train_df, test_df], ignore_index=True)

    test_data = {
        key: pd.read_csv(os.path.join(cfg.root_path, 'test', filename), index_col=0)
        for key, filename in cfg.test_data.items()
    }

    return train_data, test_data


def create_models(cfg: DictConfig) -> tuple[object, object]:
    """Create estimator and adaptation model from configuration."""
    # Create estimator
    if cfg.classifier._target_ == "sklearn.neighbors.KNeighborsClassifier":
        estimator = hydra.utils.instantiate(cfg.classifier)
    else:
        estimator = hydra.utils.instantiate(cfg.classifier, random_state=cfg.random_state)

    # Create adaptation model
    if cfg.adapt_model._target_ == "adapt.instance_based.WANN":
        adapt_model = hydra.utils.instantiate(cfg.adapt_model, random_state=cfg.random_state)
    else:
        adapt_model = hydra.utils.instantiate(
            cfg.adapt_model, estimator=estimator, random_state=cfg.random_state
        )

    return estimator, adapt_model


def prepare_wandb_config(cfg: DictConfig) -> Dict:
    """Convert config to dict and filter out non-serializable keys for wandb."""
    config_dict = {}
    for key, value in cfg.items():
        try:
            json.dumps({key: value})
            config_dict[key] = value
        except (TypeError, ValueError):
            continue
    return config_dict


def create_visualizations(cfg: DictConfig, test_data: Dict[str, pd.DataFrame], adapted_data: Dict) -> None:
    """Create and save adaptation visualizations."""
    if not cfg.get('create_visualizations', True):
        return

    print("Creating adaptation visualizations...")
    viz = AdaptationVisualizer()
    viz_dir = os.path.join("outputs", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load original test data for comparison
    original_test_data = {
        key: pd.read_csv(os.path.join(cfg.root_path, 'test', filename), index_col=0)
        for key, filename in cfg.test_data.items()
    }

    # Get numeric parameters (limit to 6 for readability)
    sample_df = list(test_data.values())[0]
    blood_params = [
        col for col in sample_df.columns
        if col != 'target' and pd.api.types.is_numeric_dtype(sample_df[col])
    ][:6]

    countries = list(cfg.test_data.keys())

    # Create distribution comparison
    viz.plot_distribution_comparison(
        original_test_data, adapted_data,
        countries=countries, blood_params=blood_params,
        save_path=os.path.join(viz_dir, "distribution_comparison.png")
    )

    # Create scatter plot comparison (if we have at least 2 parameters)
    if len(blood_params) >= 2:
        viz.plot_scatter_comparison(
            original_test_data, adapted_data,
            x_param=blood_params[0], y_param=blood_params[1],
            countries=countries,
            save_path=os.path.join(viz_dir, "scatter_comparison.png")
        )

    # Create statistical comparison
    stats_df = viz.plot_statistics_comparison(
        original_test_data, adapted_data,
        countries=countries, blood_params=blood_params,
        save_path=os.path.join(viz_dir, "statistics_comparison.png")
    )

    # Save statistics to CSV
    stats_df.to_csv(os.path.join(viz_dir, "adaptation_statistics.csv"), index=False)
    print(f"Visualizations saved to: {viz_dir}")


def prepare_metrics(cfg: DictConfig) -> List[Callable]:
    """Prepare evaluation metrics from configuration."""
    metrics = []
    for metric_config in cfg.metrics:
        if metric_config['_target_'] in ['sklearn.metrics.accuracy_score', 'sklearn.metrics.roc_auc_score']:
            metric = hydra.utils.instantiate(metric_config, _partial_=True)
        else:
            metric = hydra.utils.instantiate(metric_config, _partial_=True, zero_division=0.0)
        metrics.append(metric)
    return metrics


def run_experiment(cfg: DictConfig, adapter: Adapter, test_data: Dict[str, pd.DataFrame],
                  metrics: List[Callable]) -> float:
    """Run the complete adaptation experiment and log results."""
    # Train on adapted data and evaluate
    adapted_results = adapter.train_on_adapted_data(
        test_data=test_data,
        metrics=metrics,
        use_weights=True,
        save_model=True
    )
    wandb.log(adapted_results)

    # Calculate baseline results
    baseline_results = adapter.calc_baseline(test_data, metrics=metrics)
    wandb.log(baseline_results)

    # Compare adapted vs baseline
    comparison_results = adapter.compare(adapted_results, baseline_results, metrics, test_data)
    wandb.log(comparison_results)

    # Calculate optimization score
    metric_scores = [
        score for key, score in adapted_results.items()
        if cfg.optimized_metric in key
    ]
    mean_score = mean(metric_scores)

    print(f"Mean {cfg.optimized_metric} score: {mean_score}")
    wandb.log({f"{cfg.optimized_metric}_mean_score": mean_score})

    return mean_score


@hydra.main(version_base="1.1", config_path="experiments/config", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main function for cross-adaptation experiments."""
    # Load data
    train_data, test_data = load_datasets(cfg)

    # Create models
    estimator, adapt_model = create_models(cfg)

    # Initialize wandb
    config_dict = prepare_wandb_config(cfg)
    run_name = f"{cfg.classifier._target_}_{cfg.adapt_model._target_}_{random.randint(0, 100000)}"

    wandb.init(
        entity="thelion-ai",
        project="cross-adaptation-5",
        config=config_dict,
        name=run_name
    )

    try:
        # Create adapter and run adaptation
        adapter = Adapter(train_data=train_data, estimator=estimator, adapt_model=adapt_model)
        save_dir = os.path.join("outputs", "adapted_data")
        adapted_data = adapter.adapt(save_dir=save_dir)

        # Create visualizations
        create_visualizations(cfg, test_data, adapted_data)

        # Prepare metrics and run experiment
        metrics = prepare_metrics(cfg)
        mean_score = run_experiment(cfg, adapter, test_data, metrics)

        return mean_score

    finally:
        wandb.finish()
    
if __name__ == "__main__":
    main()
