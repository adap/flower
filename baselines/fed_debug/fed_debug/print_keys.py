"""Just printing keys from cache."""

import hydra
from diskcache import Index
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="train", version_base=None)
def _print_train_keys(cfg: DictConfig) -> None:

    cache = Index(cfg.storage.dir + cfg.storage.cache_name)
    print(f"Train Cache: {cfg.storage.dir + cfg.storage.cache_name}")
    for k in cache.keys():
        if k.find("round") == -1:
            k.replace("(", "\\(").replace(")", "\\)")
            print(f'"{k}",')

    print(f"---> Train Cache: {cfg.storage.dir + cfg.storage.cache_name}")


@hydra.main(config_path="conf", config_name="debug", version_base=None)
def _print_debug_keys(cfg: DictConfig) -> None:
    cache = Index(cfg.storage.dir + cfg.storage.debug_cache_name)
    print(f"debug Cache: {cfg.storage.dir + cfg.storage.debug_cache_name}")
    for k in cache.keys():
        print(f'"{k}",')

    print(f"---> debug Cache: {cfg.storage.dir + cfg.storage.debug_cache_name}")


if __name__ == "__main__":
    # _print_debug_keys()
    _print_train_keys()
