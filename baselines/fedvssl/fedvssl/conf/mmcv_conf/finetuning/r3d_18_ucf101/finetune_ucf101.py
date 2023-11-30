_base_ = ['../model_r3d18.py',
          '../runtime_ucf101.py']

work_dir = './finetune_results/'

model = dict(
    backbone=dict(
        pretrained='./model_pretrained.pth',
    ),
)
