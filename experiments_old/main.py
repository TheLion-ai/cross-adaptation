import random
import os
from statistics import mean

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from cross_adaptation import Adapter


@hydra.main(
    version_base="1.1", config_path="../experiments/config", config_name="config"
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

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=cfg,
        name=f"{cfg.classifier._target_}_{cfg.adapt_model._target_}_{random.randint(0, 100000)}",
    )
    # Run cross-adaptation
    adapter = Adapter(
        train_data=train_data, estimator=estimator, adapt_model=adapt_model
    )
    transformed_data = adapter.adapt()
    for dataset_name, df in transformed_data.items():
        table = wandb.Table(dataframe=df)
        run.log({f"{dataset_name}_cross_adapted": table})

    # Test on the adapted data
    metrics = [
        hydra.utils.instantiate(metric, _partial_=True, zero_division=0.0) \
        if metric['_target_'] not in ['sklearn.metrics.accuracy_score', 'sklearn.metrics.roc_auc_score'] \
        else hydra.utils.instantiate(metric, _partial_=True) \
        for metric in cfg.metrics
    ]
    results = adapter.test(test_data, metrics=metrics)
    wandb.log(results)

    baseline_results = adapter.calc_baseline(test_data, metrics=metrics)
    wandb.log(baseline_results)
    compare_results = adapter.compare(results, baseline_results, metrics, test_data)
    wandb.log(compare_results)
    # Calculate the score for optuna
    score = dict(
        filter(lambda item: cfg.optimized_metric in item[0], results.items())
    ).values()
    score = mean(list(score))
    wandb.log({f"{cfg.optimized_metric}_mean_score": score})
    # Return the score for the optimizer
    wandb.finish()
    return score


main()
