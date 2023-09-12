dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True
#CUDA_VISIBLE_DEVICES='4,5'
data = dict(
    train=dict(
        type='CtPDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
    )
)

# data = dict(
#     videos_per_gpu=4,  # total batch size is 8Gpus*4 == 32
#     workers_per_gpu=4,
#     train=dict(
#         type='CtPDataset',
#         data_source=dict(
#             type='JsonClsDataSource',
#             ann_file='',
#         ),
#         backend=dict(
#             type='ZipBackend',
#             zip_fmt='zips/{}.zip',
#             frame_fmt='img_{:05d}.jpg',
#         ),
#         frame_sampler=dict(
#             type='RandomFrameSampler',
#             num_clips=1,
#             clip_len=16,
#             strides=[1, 2, 3, 4, 5],
#             temporal_jitter=True
#         ),
#         transform_cfg=[
#             dict(type='GroupScale', scales=[112, 128, 144]),
#             dict(type='GroupRandomCrop', out_size=112),
#             dict(type='GroupFlip', flip_prob=0.50),
#             dict(
#                 type='PatchMask',
#                 region_sampler=dict(
#                     scales=[16, 24, 28, 32, 48, 64],
#                     ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
#                     scale_jitter=0.18,
#                     num_rois=3,
#                 ),
#                 key_frame_probs=[0.5, 0.3, 0.2],
#                 loc_velocity=3,
#                 size_velocity=0.025,
#                 label_prob=0.8
#             ),
#             dict(type='RandomHueSaturation', prob=0.25, hue_delta=12, saturation_delta=0.1),
#             dict(type='DynamicBrightness', prob=0.5, delta=30, num_key_frame_probs=(0.7, 0.3)),
#             dict(type='DynamicContrast', prob=0.5, delta=0.12, num_key_frame_probs=(0.7, 0.3)),
#             dict(
#                 type='GroupToTensor',
#                 switch_rgb_channels=True,
#                 div255=True,
#                 mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225)
#             ),
#         ]
#     ),
#     val=dict(
#         type='CtPDataset',
#         data_source=dict(
#             type='JsonClsDataSource',
#             ann_file='ucf101/annotations/test_split_1.json',
#         ),
#         backend=dict(
#             type='ZipBackend',
#             zip_fmt='ucf101/zips/{}.zip',
#             frame_fmt='img_{:05d}.jpg',
#         ),
#         frame_sampler=dict(
#             type='RandomFrameSampler',
#             num_clips=1,
#             clip_len=16,
#             strides=[1, 2, 3, 4, 5],
#             temporal_jitter=True
#         ),
#         transform_cfg=[
#             dict(type='GroupScale', scales=[112, 128, 144]),
#             dict(type='GroupRandomCrop', out_size=112),
#             dict(
#                 type='PatchMask',
#                 region_sampler=dict(
#                     scales=[16, 24, 28, 32, 48, 64],
#                     ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
#                     scale_jitter=0.18,
#                     num_rois=3,
#                 ),
#                 key_frame_probs=[0.5, 0.3, 0.2],
#                 loc_velocity=3,
#                 size_velocity=0.025,
#                 label_prob=0.8
#             ),
#             dict(
#                 type='GroupToTensor',
#                 switch_rgb_channels=True,
#                 div255=True,
#                 mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225)
#             ),
#         ]
#     )
# )

# optimizer
total_epochs = 1
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[100, 200]
)
checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
