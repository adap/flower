"""Config file used for fine-tuning on UCF-101 dataset."""

_base_ = [
    "../../recognizers/_base_/model_r3d18.py",
    "../../recognizers/_base_/runtime_ucf101.py",
]

work_dir = "./output/ctp/r3d_18_ucf101/finetune_ucf101/"

model = {
    "backbone": {
        "pretrained": "./output/ctp/r3d_18_ucf101/pretraining/epoch_300.pth",
    },
}
