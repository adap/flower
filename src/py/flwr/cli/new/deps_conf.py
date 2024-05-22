"""Module defining dpendency lists for different frameworks."""

deps = {
    "pytorch": """[
  "flwr[simulation]>=1.8.0,<2.0",
  "flwr-datasets[vision]>=0.0.2,<1.0.0",
  "torch==2.2.1",
  "torchvision==0.17.1",
]""",
    "hf": """[
  "flwr[simulation]>=1.8.0,<2.0"
  "flwr-datasets>=0.0.2,<1.0.0"
  "torch==2.2.1"
  "transformers>=4.30.0,<5.0"
  "evaluate>=0.4.0,<1.0"
  "datasets>=2.0.0, <3.0"
  "scikit-learn>=1.3.1, <2.0",
]""",
    "mlx": """[
  "flwr[simulation]>=1.8.0,<2.0",
  "flwr-datasets[vision]>=0.0.2,<1.0.0",
  "mlx==0.10.0",
  "numpy==1.24.4",
]""",
    "sklearn": """[
  "flwr[simulation]>=1.8.0,<2.0",
  "flwr-datasets[vision]>=0.0.2,<1.0.0",
  "scikit-learn>=1.1.1"
]""",
    "tensorflow": """[
  "flwr[simulation]>=1.8.0,<2.0",
  "flwr-datasets[vision]>=0.0.2,<1.0.0",
  "tensorflow>=2.11.1",
]""",
    "numpy": """[
  "flwr[simulation]>=1.8.0,<2.0",
  "numpy>=1.21.0",
]""",
}
