import hydra
from omegaconf import OmegaConf, DictConfig
import train


import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


@hydra.main(version_base=None, config_path="configs", config_name="local")
def main(c: DictConfig):
    OmegaConf.set_struct(c, False)
    train.train_and_evaluate(c)


if __name__ == "__main__":
    main()
