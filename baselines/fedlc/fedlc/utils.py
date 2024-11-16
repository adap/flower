"""fedlc: A Flower Baseline."""


def gen_checkpoint_suffix(alg: str, dataset: str, dirichlet_alpha: float):
    return f"{alg}_dataset={dataset.split('/')[-1]}_alpha={dirichlet_alpha}_r="
