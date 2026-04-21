# show_config.py

from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).parent / "config")

def load_config(overrides=None):
    """Compose the project's Hydra configuration."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides or [])

if __name__ == "__main__":
    import sys
    # Anything after "--" is a list of Hydra overrides.
    # e.g. python show_config.py -- training.cv_folds=10 model=linear
    overrides = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    print(OmegaConf.to_yaml(load_config(overrides)))

