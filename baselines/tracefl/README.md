---

title: TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance
url: [https://arxiv.org/abs/2312.13632](https://arxiv.org/abs/2312.13632)
labels: \[interpretability, debugging, federated-learning, neuron-provenance, explainability, computer-vision, nlp, dirichlet-partitioning, client-attribution, auditing]
dataset: \[MNIST, CIFAR-10, PathMNIST, YahooAnswers]
------------------------------------------------------------------

> \[!NOTE]
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance](https://arxiv.org/abs/2312.13632)

**Authors:** Waris Gill (Virginia Tech), Ali Anwar (University of Minnesota Twin Cities), Muhammad Ali Gulzar (Virginia Tech)

**Abstract:** Federated Learning (FL) enables decentralized model training but introduces a major challenge: attributing a global model's predictions to specific clients. This limitation makes debugging, rewarding useful participants, and detecting faulty ones difficult. We propose TraceFL, the first interpretability technique for FL that performs neuron-level provenance tracking. TraceFL identifies clients responsible for specific predictions by tracking activated neurons and mapping them to local models. We evaluate it on six datasets (image and text) using a variety of neural architectures. TraceFL achieves up to 99% localization accuracy, enabling fine-grained FL debugging and auditing.

## About this baseline

**What’s implemented:** This Flower baseline replicates the core experiments from the TraceFL paper, including:

* Figure 2, Figure 5, and Table 3 (Correct Client Attribution)
* Figure 3 (Impact of Dirichlet Alpha on Attribution)

**Datasets:**

* MNIST (60k samples)
* CIFAR-10 (50k samples)
* PathMNIST (89k samples)
* YahooAnswers (public NLP datasets)

All datasets are partitioned using Dirichlet distribution (alpha configurable). FlowerDatasets API is used for loading and partitioning.

**Hardware Setup:**

* Recommended: 1x GPU (>=16GB VRAM), 32GB RAM

**Contributors:** Ibrahim Ahmed Khan (iak.ibrahimkhan@gmail.com)

## Experimental Setup

**Task:**

* Image classification (MNIST, CIFAR-10, PathMNIST)
* Text classification (AGNews, IMDB, YahooAnswers)

**Model:**

* Vision: ResNet18, ResNet50, DenseNet121
* NLP: DistilBERT, TinyBERT, GPT, MiniLM

**Dataset Partitioning:**

| Dataset      | Partitioning          | Clients | Classes per Client | Partition Method    |
| ------------ | --------------------- | ------- | ------------------ | ------------------- |
| MNIST        | Dirichlet (alpha=0.3) | 10      | 2-3                | non\_iid\_dirichlet |
| CIFAR-10     | Dirichlet (alpha=0.3) | 10      | 2-3                | non\_iid\_dirichlet |
| PathMNIST    | Dirichlet (alpha=0.3) | 10      | 2-3                | non\_iid\_dirichlet |
| AGNews       | Natural               | 4       | All                | pre-partitioned     |
| IMDB         | Natural               | 2       | All                | pre-partitioned     |
| YahooAnswers | Natural               | 10      | All                | pre-partitioned     |

**Training Hyperparameters:**

| Parameter         | Value |
| ----------------- | ----- |
| Learning Rate     | 0.001 |
| Batch Size        | 32    |
| Rounds            | 2     |
| Local Epochs      | 2     |
| Clients per Round | 4     |
| Optimizer         | Adam  |

## Environment Setup

```bash
# Create the virtual environment
pyenv virtualenv 3.10.14 tracefl

# Activate it
pyenv activate tracefl

# Install the baseline
pip install -e .
```

## Running the Experiments

### Experiment 1 — Correct Prediction Tracing 

```bash
flwr run .
```
or 
```bash
flwr run . --run-config exp_1
```

### Experiment 2 — Varying Dirichlet Alpha

```bash
flwr run . --run-config exp_2
```

## 📎 Additional Resources

* 🔗 [Original TraceFL Implementation in Colab (Full Paper Reproduction)](https://colab.research.google.com/github/SEED-VT/TraceFL/blob/main/reproducing.ipynb#scrollTo=k6IPOmpcGfLp)
* 🧪 Terminal results are also logged in `TraceFL_clients_contributions.log`

## Citation

```bibtex
@inproceedings{tracefl2025,
  title={TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance},
  author={Gill, Waris and Anwar, Ali and Gulzar, Muhammad Ali},
  booktitle={International Conference on Software Engineering (ICSE)},
  year={2025}
}

@article{flower2020,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel and Topal, T and Qiu, X and others},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

---

> Maintained as part of the Flower Baselines Collection
