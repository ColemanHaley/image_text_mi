import logging

import hydra
import numpy as np
import pandas as pd
import torch
from pandas.core.common import flatten
from PIL import Image
from rich import traceback
from rich.logging import RichHandler
from torch.distributions import Categorical
from torch.utils.data import DataLoader

# from tqdm.rich import tqdm
from tqdm import tqdm
logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

LEN_CAPTIONING_PREFIX = 2  # TODO: compute prefix length

traceback.install()


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):


    hydra_output = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{hydra_output}/{cfg.out_file}", "w") as f:
        f.write("") 
    with open(f"{hydra_output}/../../{cfg.out_file}", "w") as f:
        f.write("helloooo")
    # with open("outputs/{cfg.out_file}", "w") as f:
    #     pd.concat(full_results, ignore_index=True).to_csv(f)


if __name__ == "__main__":
    main()
