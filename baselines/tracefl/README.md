---
title: TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance
url: https://arxiv.org/abs/2312.13632
labels: [interpretability, neuron provenance, federated learning, debugging, non-iid]
dataset: [MNIST, CIFAR-10, CIFAR-100, PathMNIST, OrganAMNIST, DBpedia-14, Yahoo Answers Topics]
---

# TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/2312.13632](https://arxiv.org/abs/2312.13632)

**Authors:** Waris Gill, Ali Anwar, Muhammad Ali Gulzar

**Abstract:** TraceFL is the first interpretability technique that enables interpretability in Federated Learning (FL) by identifying clients responsible for specific global model predictions. By making such provenance information explicit, developers can exclude problematic clients, reward high-quality clients, or debug misclassifications more systematically. TraceFL uses neuron-level provenance tracking to trace client contributions to global model predictions, providing interpretability in FL systems.

## About this baseline

**What's implemented:** This Flower baseline implements TraceFL, a federated learning interpretability framework that identifies clients responsible for specific global model predictions through neuron provenance analysis. The implementation replicates key experiments from the original paper including localization accuracy evaluation, data distribution analysis, faulty client detection, and differential privacy experiments. It supports multiple datasets (MNIST, CIFAR-10, CIFAR-100, PathMNIST, OrganAMNIST, DBpedia-14, Yahoo Answers Topics) and model architectures (ResNet, CNN, Transformer).

**Datasets:** MNIST (60,000 training, 10,000 test), CIFAR-10 (50,000 training, 10,000 test), CIFAR-100 (50,000 training, 10,000 test), PathMNIST (107,180 training, 7,180 test), OrganAMNIST (34,581 training, 17,778 test), DBpedia-14 (560,000 training, 70,000 test), Yahoo Answers Topics (1,400,000 training, 60,000 test)

**Hardware Setup:** This baseline was tested on a desktop machine with 8 CPU cores and 32GB RAM. Experiments run efficiently on CPU-only mode with minimal resource requirements. The baseline supports both CPU and GPU execution, with GPU acceleration available for larger models. Minimum requirements: 4 CPU cores, 8GB RAM for basic experiments.

**Contributors:** Ibrahim A. Alzahrani

## Experimental Setup

**Task:** Multi-domain classification (image classification, medical imaging, text classification)

**Model:** The baseline supports multiple model architectures:
- **Image Classification**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, DenseNet121, CNN
- **Text Classification**: DistilBERT, GPT-based models
- **Medical Imaging**: ResNet and DenseNet adapted for grayscale inputs

**Dataset:** The baseline uses non-IID Dirichlet distribution for data partitioning across clients. The `dirichlet_alpha` parameter controls the degree of non-IIDness:
- α → ∞: All clients have identical distribution (IID)
- α → 0: Each client holds samples from only one class (extreme non-IID)

| Dataset | # Classes | # Clients | Partition Method | Default α |
| :------ | :-------: | :-------: | :--------------: | :-------: |
| MNIST | 10 | 10 | Dirichlet | 0.3 |
| CIFAR-10 | 10 | 10 | Dirichlet | 0.3 |
| CIFAR-100 | 100 | 10 | Dirichlet | 0.3 |
| PathMNIST | 9 | 10 | Dirichlet | 0.3 |
| OrganAMNIST | 11 | 10 | Dirichlet | 0.3 |
| DBpedia-14 | 14 | 10 | Dirichlet | 0.3 |
| Yahoo Answers Topics | 10 | 10 | Dirichlet | 0.3 |

**Training Hyperparameters:**

| Description | Default Value |
| ----------- | ------------- |
| Number of clients | 10 |
| Number of rounds | 2-3 |
| Model | ResNet18 |
| Dataset | MNIST |
| Dirichlet alpha | 0.3 |
| Batch size | 32 |
| Max per-client data size | 2048 |
| Max server data size | 2048 |
| Provenance rounds | 1,2 |
| Use deterministic sampling | true |
| Random seed | 42 |
| Min train nodes | 4 |
| Fraction train | 0.4 |

## Environment Setup

This baseline uses Python 3.10, pyenv, and virtualenv. Follow these steps to set up the environment:

```bash
# Create the virtual environment
pyenv virtualenv 3.10.14 tracefl-baseline

# Activate it
pyenv activate tracefl-baseline

# Install the baseline
pip install -e .
```

## Running the Experiments

The baseline provides organized experiment scripts that save results in separate folders for easier navigation:

### Experiment A: Localization Accuracy (Figure 2, Table 3, Figure 5)
```bash
bash scripts/a_figure_2_table_3_and_figure_5_single_alpha.sh
```
This runs TraceFL with Dirichlet alpha=0.3 on MNIST with ResNet18, evaluating localization accuracy for correct predictions.

**Results saved to:** `results/experiment_a/`

### Experiment B: Data Distribution Analysis (Figure 3)
```bash
bash scripts/b_figure_3.sh
```
This runs TraceFL with different Dirichlet alpha values to analyze the effect of data distribution on localization accuracy.

**Results saved to:** `results/experiment_b/`

### Experiment C: Faulty Client Detection (Table 1, Figure 6)
```bash
bash scripts/c_table_1_and_figure_6.sh
```
This runs TraceFL with simulated faulty clients that flip labels to evaluate localization accuracy for mispredictions.

**Results saved to:** `results/experiment_c/`

### Experiment D: Differential Privacy (Figure 4, Table 2)
```bash
bash scripts/d_figure_4_and_table_2.sh
```
This runs TraceFL with differential privacy enabled (noise multiplier=0.001, clipping norm=15) to evaluate provenance analysis under privacy constraints.

**Results saved to:** `results/experiment_d/`

### Custom Experiments
You can also run custom experiments by overriding configuration parameters:

```bash
# Run with different dataset and model
flwr run . --run-config "tracefl.dataset='cifar10' tracefl.model='resnet50' tracefl.dirichlet-alpha=0.1"

# Run with differential privacy
flwr run . --run-config "tracefl.noise-multiplier=0.01 tracefl.clipping-norm=10"

# Run with different number of clients
flwr run . --run-config "tracefl.num-clients=20 min-train-nodes=8"
```

## Expected Results

### Localization Accuracy
TraceFL achieves high localization accuracy (>90%) for correct predictions across different data distributions. The accuracy decreases with more extreme non-IID distributions (lower alpha values).

### Faulty Client Detection
TraceFL can effectively identify faulty clients that contribute to mispredictions, with localization accuracy >80% even when clients flip labels.

### Differential Privacy Impact
When differential privacy is enabled, TraceFL maintains reasonable localization accuracy (>70%) while providing privacy guarantees.

### Generated Outputs
Each experiment generates:
- **Provenance CSV files**: Detailed client contributions for each prediction
- **Accuracy plots**: Visualization of localization accuracy over rounds
- **Final model**: Trained global model saved as `final_model.pt`
- **Summary tables**: Aggregated accuracy statistics

## Key Features

- **Neuron-level provenance tracking**: Traces individual neuron contributions to predictions
- **Multi-domain support**: Works with image, text, and medical imaging datasets
- **Faulty client simulation**: Simulates malicious clients for robustness evaluation
- **Differential privacy integration**: Evaluates provenance under privacy constraints
- **Organized output structure**: Results saved in experiment-specific folders
- **Reproducible experiments**: Deterministic sampling with configurable random seeds

## Citation

If you use this baseline in your research, please cite the original TraceFL paper:

```bibtex
@inproceedings{gill2025tracefl,
  title = {{TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance}},
  author = {Gill, Waris and Anwar, Ali and Gulzar, Muhammad Ali},
  booktitle = {2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)},
  year = {2025},
  organization = {IEEE},
}
```
