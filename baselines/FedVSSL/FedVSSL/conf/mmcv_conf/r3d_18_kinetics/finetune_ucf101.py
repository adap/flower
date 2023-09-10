_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_ucf101.py']

work_dir = './output/ctp/r3d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        pretrained='./output/ctp/r3d_18_kinetics/pretraining/epoch_90.pth',
    ),
)
