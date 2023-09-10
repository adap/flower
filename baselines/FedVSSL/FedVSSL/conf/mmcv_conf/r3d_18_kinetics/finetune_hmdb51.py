_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_hmdb51.py']

work_dir = './output/ctp/r3d_18_kinetics/finetune_hmdb51/'

model = dict(
    backbone=dict(
        pretrained='./output/ctp/r3d_18_kinetics/pretraining/epoch_90.pth',
    ),
    cls_head=dict(
        num_classes=51
    )
)