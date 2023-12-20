"""Config file used for pre-training on UCF-101 dataset."""

dist_params = {"backend": "nccl"}
log_level = "INFO"
load_from = None
resume_from = None
syncbn = True
# CUDA_VISIBLE_DEVICES='4,5'

data = {
    "videos_per_gpu": 4,  # total batch size is 8Gpus*4 == 32
    "workers_per_gpu": 4,
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
        "frame_sampler": {
            "type": "RandomFrameSampler",
            "num_clips": 1,
            "clip_len": 16,
            "strides": [1, 2, 3, 4, 5],
            "temporal_jitter": True,
        },
        "transform_cfg": [
            {"type": "GroupScale", "scales": [112, 128, 144]},
            {"type": "GroupRandomCrop", "out_size": 112},
            {"type": "GroupFlip", "flip_prob": 0.50},
            {
                "type": "PatchMask",
                "region_sampler": {
                    "scales": [16, 24, 28, 32, 48, 64],
                    "ratios": [0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                    "scale_jitter": 0.18,
                    "num_rois": 3,
                },
                "key_frame_probs": [0.5, 0.3, 0.2],
                "loc_velocity": 3,
                "size_velocity": 0.025,
                "label_prob": 0.8,
            },
            {
                "type": "RandomHueSaturation",
                "prob": 0.25,
                "hue_delta": 12,
                "saturation_delta": 0.1,
            },
            {
                "type": "DynamicBrightness",
                "prob": 0.5,
                "delta": 30,
                "num_key_frame_probs": (0.7, 0.3),
            },
            {
                "type": "DynamicContrast",
                "prob": 0.5,
                "delta": 0.12,
                "num_key_frame_probs": (0.7, 0.3),
            },
            {
                "type": "GroupToTensor",
                "switch_rgb_channels": True,
                "div255": True,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        ],
    },
    "val": {
        "type": "CtPDataset",
        "data_source": {
            "type": "JsonClsDataSource",
            "ann_file": "annotations/test_split_1.json",
        },
        "backend": {
            "type": "ZipBackend",
            "zip_fmt": "zips/{}.zip",
            "frame_fmt": "img_{:05d}.jpg",
        },
        "frame_sampler": {
            "type": "RandomFrameSampler",
            "num_clips": 1,
            "clip_len": 16,
            "strides": [1, 2, 3, 4, 5],
            "temporal_jitter": True,
        },
        "transform_cfg": [
            {"type": "GroupScale", "scales": [112, 128, 144]},
            {"type": "GroupRandomCrop", "out_size": 112},
            {
                "type": "PatchMask",
                "region_sampler": {
                    "scales": [16, 24, 28, 32, 48, 64],
                    "ratios": [0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                    "scale_jitter": 0.18,
                    "num_rois": 3,
                },
                "key_frame_probs": [0.5, 0.3, 0.2],
                "loc_velocity": 3,
                "size_velocity": 0.025,
                "label_prob": 0.8,
            },
            {
                "type": "GroupToTensor",
                "switch_rgb_channels": True,
                "div255": True,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        ],
    },
}

# optimizer
total_epochs = 1
optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001}
optimizer_config = {"grad_clip": {"max_norm": 40, "norm_type": 2}}
# learning policy
lr_config = {"policy": "step", "step": [100, 200]}
checkpoint_config = {"interval": 1, "max_keep_ckpts": 1, "create_symlink": False}
workflow = [("train", 1)]
log_config = {
    "interval": 10,
    "hooks": [
        {"type": "TextLoggerHook"},
        {"type": "TensorboardLoggerHook"},
    ],
}
