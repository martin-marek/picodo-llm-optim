import hydra
import train, utils
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="local")
def main(c: DictConfig):
  OmegaConf.set_struct(c, False)
  print(utils.flatten_dict(c))
  train.train_and_evaluate(c)

if __name__ == "__main__":
  main()