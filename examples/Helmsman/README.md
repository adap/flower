# ☸ Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents

<div align="center">

<div>
    <a href="https://haoyuan-l.github.io/" target="_blank">Haoyuan Li</a><sup>1</sup>&emsp;
    <a href="https://mathias-funk.com/" target="_blank">Mathias Funk</a><sup>1</sup>&emsp;
    <a href="https://aqibsaeed.github.io/" target="_blank">Aaqib Saeed</a><sup>1</sup>&emsp;
</div>

<div>
    <sup>1</sup><a href="https://www.tue.nl/en/our-university/departments/industrial-design/research/our-research-labs/decentralized-artificial-intelligence-research-lab" target="_blank" rel="noopener noreferrer">
    Decentralized Artificial Intelligence Research Lab, Eindhoven University of Technology
    </a>
</div>

</div>

## ☸ Helmsman

**Helmsman** is a pioneering multi-agent framework designed to automate the end-to-end synthesis of Federated Learning (FL) systems. By bridging the gap between high-level user intent and executable, robust code, Helmsman addresses the intractable complexity of the FL design space.

This app hosts the implementation of the paper **"Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents"** from the [official repository](https://github.com/haoyuan-l/Helmsman), published at **ICLR 2026**.

---

## 📖 Introduction

Federated Learning (FL) holds immense promise for privacy-centric collaborative AI, yet its practical deployment remains a complex engineering challenge. Designing an effective FL system requires navigating a combinatorial design space defined by statistical heterogeneity, system constraints, and shifting task objectives. To date, this process has been a manual, labor-intensive effort led by domain experts, resulting in bespoke solutions that are often brittle in the face of real-world dynamics.

To bridge this gap, we introduce **Helmsman**, a multi-agent system designed to automate the end-to-end research and development of task-oriented FL systems. Helmsman moves beyond simple code generation to holistic system synthesis, navigating the intractable design space by emulating a principled R&D workflow.

**Key Contributions:**

* **End-to-End Synthesis**: We develop **Helmsman**, an agentic framework that translates high-level specifications into deployable FL systems through three collaborative phases: *Interactive Planning*, *Modular Coding*, and *Autonomous Evaluation*.
* **Novel Benchmark**: We introduce **AgentFL-Bench**, a rigorous benchmark comprising 16 diverse tasks across 5 research areas, designed to evaluate the system-level generation capabilities of agentic systems.
* **SOTA Performance**: Extensive experiments demonstrate that Helmsman-generated solutions achieve performance competitive with, and often exceeding, established hand-crafted FL baselines.

---

## 🏗️ System Architecture

Helmsman orchestrates a collaborative team of LLM agents through a principled, three-phase R&D workflow.

### 1. Interactive Planning
A **Planning Agent** synthesizes a rigorous research plan using external web search and an internal **RAG pipeline** of FL literature. A **Reflection Agent** critiques the strategy for theoretical validity, followed by a final **Human-in-the-Loop** verification to ensure technical soundness and alignment.

### 2. Modular Code Generation
A **Supervisor Agent** decomposes the plan into a modular blueprint (*Task, Client, Strategy, Server*). Specialized **Coder** and **Tester** teams then implement and verify these components in parallel, ensuring separation of concerns and robust code quality.

### 3. Autonomous Refinement
To guarantee robustness, the system runs a closed-loop **Flower simulation**. An **Evaluator Agent** diagnoses runtime and semantic failures, triggering a **Debugger Agent** to iteratively patch the code until the system is certified as fully executable and high-performing.

---

## ⚙️ Installation

### Prerequisites
* OS: Linux (Recommended)
* Python: 3.12
* CUDA: 12.9 (for GPU acceleration)

### 1. Environment Setup
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage the environment.

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create and activate the environment
conda create --name agenticfl python=3.12
conda activate agenticfl
```

### 2. Install Core FL Dependencies

Helmsman generates FL systems built on the following pinned framework versions:

| Package | Version | Role |
| :--- | :---: | :--- |
| [`flwr[simulation]`](https://flower.ai) | **1.20.0** | Federated learning framework and simulation engine |
| [`flwr-datasets[vision]`](https://flower.ai/docs/datasets/) | **0.5.0** | Federated dataset loading and partitioning |

```bash
# Install PyTorch (CUDA 12.9)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install Flower framework and datasets at the pinned versions
pip install "flwr[simulation]==1.20.0"
pip install "flwr-datasets[vision]==0.5.0"

# Audio processing support
pip install librosa
pip install soundfile
```

### 3. Install Agent Framework
Install LangChain, LangGraph, and provider packages for the LLM agents.
```bash
pip install python-dotenv
pip install tree-sitter
pip install -U langchain
pip install -U langchain-openai
pip install -U langchain-anthropic
pip install -U langchain-google-genai
pip install -U langgraph
pip install huggingface-hub
```

---

## 🔑 API Configuration

Helmsman requires API keys from various LLM providers. We recommend using a `.env` file to manage your keys securely.

**1. Create a `.env` file in the project root directory.**

**2. Edit `.env` and add your API keys:**

```env
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Utility APIs (required for tools)
VOYAGE_API_KEY=your_voyage_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 🚀 Usage

To start Helmsman, execute the main script. The system will prompt you to select models and confirm API keys.

```bash
python agenticFL_workflow.py
```

### 1. API Setup
Select your LLM models and confirm API keys when prompted.

### 2. Input Query
Provide a natural language description of your FL experiment when asked.

**Example Query:**

> "I need to deploy a personalized handwriting recognition app across 15 mobile devices. Each client holds FEMNIST data from individual users with unique writing styles. Help me build a personalized federated learning framework that balances global knowledge with local user adaptation for a CNN model, evaluating performance by average client test accuracy."

### 3. Plan Approval
The agents will generate a research plan. Review it and type `yes` to proceed, or provide feedback to refine it.

### 4. Auto-Coding
Helmsman will generate the code, run module-level tests, execute a simulation, and package the result.

### 5. Results
Upon success, the generated FL system is written to `fl_codebase/`. See below for how to run it.

---

## 📁 Generated Codebase Structure

After a successful run, Helmsman produces a self-contained, standards-compliant FL project at `fl_codebase/`:

```
fl_codebase/
├── application/
│   ├── __init__.py       # Package marker (auto-generated)
│   ├── client_app.py     # FL client definition (FlowerClient + ClientApp)
│   ├── server_app.py     # FL server configuration (ServerApp)
│   ├── strategy.py       # Custom aggregation strategy
│   ├── task.py           # Model architecture, data loading, train/test functions
│   └── run.py            # Simulation entry-point for standalone execution
└── pyproject.toml        # Flower project config (CLI entry-points + hyperparameters)
```

The `application/` directory is a valid Python package and `pyproject.toml` declares the Flower CLI entry-points, federation topology, and training hyperparameters. This dual structure means the generated system supports **two independent execution modes** with no manual edits required.

---

## ▶️ Running the Generated FL System

### Option 1 — Flower CLI

The standard Flower deployment path. The Flower CLI reads `pyproject.toml` to locate the `ClientApp` and `ServerApp`, resolve the federation topology, and apply all hyperparameters.

```bash
cd fl_codebase/
flwr run .
```

This mode is recommended for production use, reproducible benchmarking, or multi-machine federation. Training hyperparameters such as number of rounds, fraction of clients evaluated, local epochs, and batch size can be adjusted directly in `pyproject.toml` under `[tool.flwr.app.config]` — no Python edits needed.

**Example `pyproject.toml` structure generated by Helmsman:**

```toml
[project]
name = "fl-codebase"
version = "1.0.0"
description = "Federated Learning system generated by Helmsman"
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.20.0",
    "flwr-datasets[vision]>=0.5.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]

[tool.flwr.app]
publisher = "helmsman"

[tool.flwr.app.components]
serverapp = "application.server_app:server_app"
clientapp = "application.client_app:client_app"

[tool.flwr.app.config]
num-server-rounds = 10
fraction-evaluate = 0.5
local-epochs = 1
batch-size = 32

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 15
```

### Option 2 — Python Simulation Engine

A self-contained script mode ideal for **Google Colab**, **Jupyter notebooks**, and local experimentation — no Flower CLI installation required.

```bash
# From the project root
python fl_codebase/application/run.py

# Or from inside fl_codebase/
cd fl_codebase/
python application/run.py
```

`run.py` uses `flwr.simulation.run_simulation()` internally and resolves all imports automatically, so it works correctly regardless of the working directory.

### Compatibility Summary

| Mode | Command | Best For |
| :--- | :--- | :--- |
| **Flower CLI** | `flwr run .` (from `fl_codebase/`) | Production, benchmarking, multi-machine federation |
| **Python script** | `python application/run.py` | Google Colab, Jupyter, quick local experiments |

> **Note:** Both modes execute the exact same code in `application/`. No modifications to any source file are needed to switch between them.

---

## 📈 Key Results

We rigorously evaluated **Helmsman** on **AgentFL-Bench**, comparing its synthesized solutions against standard baselines (FedAvg, FedProx) and specialized state-of-the-art methods (e.g., FedNova, HeteroFL, FedPer).

### 1. Robustness Against Heterogeneity & Constraints
Helmsman consistently generates effective hybrid strategies that address data imbalances, distribution shifts, and system constraints, often outperforming both general and specialized baselines.

| Challenge Category | Task Examples | Helmsman Performance vs. Baselines |
| :--- | :--- | :--- |
| **Data Heterogeneity** | Quantity Skew, Label Noise | **Competitive**: Surpasses FedAvg/FedProx; achieves SOTA on Noisy Labels (Q3). |
| **Distribution Shift** | User/Speaker Variation, Domain Shift | **Superior**: Outperforms specialized methods in Human Activity (Q5) & Speech Recognition (Q6). |
| **System Constraints** | Resource & Bandwidth Limits | **Dominant**: Top performance in resource-constrained CIFAR-100 (Q9) & bandwidth-limited tasks (Q10-Q11). |

### 2. Advanced & Interdisciplinary Scenarios
Helmsman proves capable of solving complex, compound challenges that require sophisticated algorithmic reasoning.

* **Personalization**: In handwriting recognition (FEMNIST) and distribution skew tasks, Helmsman synthesizes solutions that effectively balance global knowledge with local adaptation, rivaling specialized personalization methods like FedPer.
* **Federated Continual Learning (FCL)**: Notably, in the challenging Split-CIFAR100 task (Q16), Helmsman synthesized a solution that **substantially outperformed** the specialized *FedWeIT* baseline (**50.95%** vs. 29.45%), demonstrating exceptional capability in mitigating catastrophic forgetting.

---

## 🏆 Evaluation & Performance

We evaluated Helmsman against state-of-the-art code synthesis pipelines — **Codex** (powered by GPT-5.1) and **Claude Code** (powered by Claude Sonnet 4.5) — across all 16 tasks in **AgentFL-Bench**.

**Key Findings:**
* ✅ **100% Success Rate**: Helmsman successfully synthesized valid, executable FL systems for every query, whereas standard coding agents failed more than half the time.
* 💰 **Cost Efficiency**: By optimizing agent coordination, Helmsman (GPT-5.1) reduces token consumption by **~13x** compared to raw Codex.
* ⚡ **Speed**: Helmsman delivers the fastest end-to-end solution synthesis.

### Performance Summary

| Framework | Backend LLM | Success Rate | Avg Cost ($) | Avg Tokens | Walltime (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Helmsman** | **GPT-5.1** | **100%** | **$0.57** | **177k** | **716** |
| **Helmsman** | Claude 4.5 | **100%** | $1.04 | 195k | 864 |
| Claude Code | Claude 4.5 | 43.8% | $1.70 | 2,233k | 1,218 |
| Codex | GPT-5.1 | 37.5% | $0.93 | 2,455k | 909 |

> **Note**: *Success Rate* indicates the percentage of tasks where the system produced fully executable code that passed both runtime and semantic verification. *Avg Cost* and *Tokens* represent the resources required to generate one complete solution.

---

## 📚 Citation

If you find **Helmsman** or **AgentFL-Bench** useful for your research, please cite our **ICLR 2026** paper:

```bibtex
@article{li2025helmsman,
  title={Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents},
  author={Li, Haoyuan and Funk, Mathias and Saeed, Aaqib},
  journal={arXiv preprint arXiv:2510.14512},
  year={2025}
}
```