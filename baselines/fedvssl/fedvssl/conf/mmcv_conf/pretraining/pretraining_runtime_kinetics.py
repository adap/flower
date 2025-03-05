"""Config file used for pre-training on K-400 dataset."""

_base_ = "./pretraining_runtime_ucf.py"

data = {
    "train": {
        "type": "CtPDataset",
        "data_source": {
            "type": "JsonClsDataSource",
            "ann_file": "",
        },
        "backend": {
            "type": "ZipBackend",
            "zip_fmt": "zips/{}.zip",
            "frame_fmt": "img_{:05d}.jpg",
        },
    }
}

# optimizer
total_epochs = 1
optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0001}
optimizer_config = {"grad_clip": {"max_norm": 40, "norm_type": 2}}
# learning policy
lr_config = {"policy": "step", "step": [30, 60]}
checkpoint_config = {"interval": 1, "max_keep_ckpts": 1, "create_symlink": False}
workflow = [("train", 1)]
log_config = {
    "interval": 50,
    "hooks": [
        {"type": "TextLoggerHook"},
        {"type": "TensorboardLoggerHook"},
    ],
}
