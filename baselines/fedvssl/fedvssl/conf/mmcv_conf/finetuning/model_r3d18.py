"""Config file used for fine-tuning on UCF-101 dataset."""

model = {
    "type": "TSN",
    "backbone": {
        "type": "R3D",
        "depth": 18,
        "num_stages": 4,
        "stem": {
            "temporal_kernel_size": 3,
            "temporal_stride": 1,
            "in_channels": 3,
            "with_pool": False,
        },
        "down_sampling": [False, True, True, True],
        "channel_multiplier": 1.0,
        "bottleneck_multiplier": 1.0,
        "with_bn": True,
        "zero_init_residual": False,
        "pretrained": None,
    },
    "st_module": {"spatial_type": "avg", "temporal_size": 2, "spatial_size": 7},
    "cls_head": {
        "with_avg_pool": False,
        "temporal_feature_size": 1,
        "spatial_feature_size": 1,
        "dropout_ratio": 0.5,
        "in_channels": 512,
        "init_std": 0.001,
        "num_classes": 101,
    },
}
