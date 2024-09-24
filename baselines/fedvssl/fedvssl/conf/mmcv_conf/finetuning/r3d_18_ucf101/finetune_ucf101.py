"""Config file used for fine-tuning on UCF-101 dataset."""

_base_ = ["../model_r3d18.py", "../runtime_ucf101.py"]

work_dir = "./finetune_results/"

model = {
    "backbone": {
        "pretrained": "./model_pretrained.pth",
    },
}
