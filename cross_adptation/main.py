import hydra
from omegaconf import DictConfig

import tools.adapt_domains as adapt_domains

@hydra.main(config_path='config/config.yaml')
def main(cfg: DictConfig) -> None:


    dfx = adapt_domains(
        data= cfg['train_data'],
        da_method= cfg['da_method'],
        classifier= cfg['classifier'],
        classifier_params= cfg['classifier_params'],
        da_params= cfg['da_params'],
        random_state= cfg['random_state']
    )

    test()