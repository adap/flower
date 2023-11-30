_base_ = ['../model_r3d18.py',
          '../runtime_ucf101.py']

work_dir = './finetune_ucf101/'

model = dict(
    backbone=dict(
        pretrained='/finetune/ucf101/epoch_150.pth',
    ),
)
