from .model import (
    AlexNetCIFAR,
    AlexNetFMNIST,
    AlexNetImageNet,
    DenseNet121,
    LeNet5,
    LSTMShakespeare,
    vgg19,
)


def check_config(cfg, attack_types):
    if "cosine" not in attack_types and "yeom" not in attack_types:
        cfg.canary = False
        cfg.noise = False
    else:
        if cfg.noise:
            if not cfg.canary and not cfg.noise:
                raise ValueError("ALERT: Canary and noise are both false!")
        else:
            cfg.canary = True
            cfg.dynamic_canary = True
            cfg.single_training = False

    if cfg.dataset == "cifar100":
        cfg.num_classes = 100
        cfg.image_label = "fine_label"
        cfg.image_name = "img"
        cfg.input_size = 32
    elif cfg.dataset == "cifar10":
        cfg.num_classes = 10
        cfg.image_label = "label"
        cfg.image_name = "img"
        cfg.input_size = 32
    elif cfg.dataset == "fmnist":
        cfg.num_classes = 10
        cfg.image_label = "label"
        cfg.image_name = "image"
        cfg.input_size = 28

    return cfg


def get_model(cfg, device):
    if cfg.architecture == "AlexNet":
        if cfg.dataset == "fmnist":
            return AlexNetFMNIST(num_classes=cfg.num_classes, device=device)
        elif cfg.dataset in ("cifar10", "cifar100"):
            return AlexNetCIFAR(num_classes=cfg.num_classes, device=device)
        else:
            return AlexNetImageNet(num_classes=cfg.num_classes, device=device)
    elif cfg.architecture == "VGG19":
        return vgg19()
    elif cfg.architecture == "LeNet5":
        return LeNet5(num_classes=cfg.num_classes, device=device)
    elif cfg.architecture == "DenseNet121":
        return DenseNet121(num_classes=cfg.num_classes)
    elif cfg.architecture == "LSTM" or cfg.dataset == "shakespeare":
        return LSTMShakespeare(
            vocab_size=cfg.vocab_size,
            num_classes=cfg.get("num_classes", cfg.vocab_size),
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
