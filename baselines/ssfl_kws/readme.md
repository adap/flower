[ICME 2023] This is an implementation of [Semi-Supervised Federated Learing for Keyword Spotting](https://arxiv.org/abs/2305.05110)
- An illustration of Semi-Supervised Federated Learning (SSFL) for Keyword Spotting (KWS).

## Requirements
```bash
conda create -f environment.yaml
```

## Instructions
 - Global hyperparameters are configured in `config.yml`
 - Use `process.py` to process exp results
 - Hyperparameters can be found at `process_control()` in utils.py 
 - run make_stats.py before executing any of the below given examples
 
## Examples
 - Train SSL for SpeechCommandsV1 dataset (TCResNet18, $N_\mathcal{S}=250$, Weak Augment: BasicAugment, Strong Augment: SpecAugment and MixAugment)
    ```bash
    python train_classifier_semi.py '+control_name="SpeechCommandsV1_tcresnet18_250_basic=basic-spec_fix"'
    ```
 - Test SSL for SpeechCommandsV2 dataset (TCResNet18, $N_\mathcal{S}=2500$, Weak Augment: BasicAugment, Strong Augment: BasicAugment, $M=100$, $C=0.1$, Non-IID ( $K=2$ ))
    ```ruby
    python test_classifier_semi.py '+control_name="SpeechCommandsV2_tcresnet18_2500_basic=basic_fix_100_0.1_non-iid-l-2"'

## Train/Test files
   - test_classifier.py
   - test_classifier_semi.py

## Hyperparameters
Datasets:
   - SpeechCommandsV1
   - SpeechCommandsV2

Models:
   - cnn
   - dscnn
   - lstm
   - mhattrnn
   - resnet9
   - resnet18
   - tcresnet9
   - tcresnet18
   - wresnet

Labeled Samples:
   - 250
   - 2500

Augment:
   - basic
   - basic=basic_spec
   - basic=basic-rand
   - basic=basic-rands
   - basic=basic-spec
   - basic=basic-spec-rands
   - With augment
      - fix
      - fix-mix

Device (M):
   - 10
   - 100

C:
   - 0.1

Data split type:
   - IID
   - Non-IID

K:
   - 2

 ## Acknowledgements
```
@article{diao2023semi,
  title={Semi-supervised federated learning for keyword spotting},
  author={Diao, Enmao and Tramel, Eric W and Ding, Jie and Zhang, Tao},
  journal={arXiv preprint arXiv:2305.05110},
  year={2023}
}
```

Original code can be found here: https://github.com/diaoenmao/Semi-Supervised-Federated-Learing-for-Keyword-Spotting