dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True

train_cfg = None
test_cfg = None
evaluation = dict(interval=10)

data = dict(
    videos_per_gpu=4,  # total batch size 8*4 == 32
    workers_per_gpu=4,
    train=dict(
        type='TSNDataset',
        name='ucf101_train_split1',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/train_split_1.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=16,
            strides=2,
            temporal_jitter=False
        ),
        test_mode=False,
        transform_cfg=[
                dict(type='GroupScale', scales=[(149, 112), (171, 128), (192, 144)]),
                dict(type='GroupFlip', flip_prob=0.35),
                dict(type='RandomBrightness', prob=0.20, delta=32),
                dict(type='RandomContrast', prob=0.20, delta=0.20),
                dict(type='RandomHueSaturation', prob=0.20, hue_delta=12, saturation_delta=0.1),
                dict(type='GroupRandomCrop', out_size=112),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
    ),
    val=dict(
        type='TSNDataset',
        name='ucf101_test_split1',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/test_split_1.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=10,
            clip_len=16,
            strides=2,
            temporal_jitter=False
        ),
        test_mode=True,
        transform_cfg=[
            dict(type='GroupScale', scales=[(171, 128)]),
            dict(type='GroupCenterCrop', out_size=112),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    ),
    test=dict(
        type='TSNDataset',
        name='ucf101_test_split1',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/test_split_1.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=10,
            clip_len=16,
            strides=2,
            temporal_jitter=False
        ),
        test_mode=True,
        transform_cfg=[
            dict(type='GroupScale', scales=[(171, 128)]),
            dict(type='GroupCenterCrop', out_size=112),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    ),
)

# optimizer
total_epochs = 150
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[60, 120]
)
checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 50)]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
