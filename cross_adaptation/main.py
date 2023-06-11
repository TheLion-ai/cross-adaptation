from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd
import numpy as np
from tools.cross_adaptation import CrossAdaptation
from adapt.instance_based import WANN
import wandb
from statistics import mean
import random

@hydra.main(version_base='1.1', config_path='../experiments/config', config_name='config')
def main(cfg):
    """Main function for cross-adaptation experiments"""
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=cfg)
    # Load data
    train_data = {k: pd.read_csv(f'{cfg.root_path}/{df}') for k, df in cfg.train_data.items()}
    test_data = {k: pd.read_csv(f'{cfg.root_path}/{df}') for k, df in cfg.test_data.items()}

    # Create estimator and adapt_model
    estimator = hydra.utils.instantiate(cfg.classifier, random_state=cfg.random_state)
    if cfg.adapt_model._target_ == 'adapt.instance_based.WANN':
        adapt_model = hydra.utils.instantiate(cfg.adapt_model, random_state=cfg.random_state)
    else:
        adapt_model = hydra.utils.instantiate(cfg.adapt_model, estimator=estimator, Xt=np.zeros((4,2)), random_state=cfg.random_state)
    
    # Run cross-adaptation
    cross_adaptation = CrossAdaptation(train_data=train_data, estimator=estimator, adapt_model=adapt_model)
    transformed_data = cross_adaptation.adapt()
    for dataset_name, df in transformed_data.items():
        table = wandb.Table(dataframe=df)
        run.log({f'{dataset_name}_xross_adapted': table})

    # Test on the adapted data
    metrics = [hydra.utils.instantiate(metric, _partial_=True) for metric in cfg.metrics]
    results = cross_adaptation.test(test_data, metrics=metrics)
    wandb.log(results)

    # TODO: Move testing to a separate function
    # Test
    # -> baseline
    # -> mean
    # TODO: add subfunction for modelling
    # TODO: move adapt method to partial
    # Compare with baseline - no adaptation
    baseline_results = cross_adaptation.calc_baseline(test_data, metrics=metrics)
    wandb.log(baseline_results)
    compare_results = cross_adaptation.compare(results, baseline_results, metrics, test_data)
    wandb.log(compare_results)
    # Calculate the score for the optimizer
    score = dict(filter(lambda item: cfg.optimized_metric in item[0], results.items())).values()
    score = mean(list(score))
    wandb.log({f'{cfg.optimized_metric}_mean_score': score})
    # Return the score for the optimizer
    return score

main()
