# Flower Benchmarks

## Structure

Each task in this directory is fully self-contained in terms of source code in its own directory. 
In addition, each task uses its very own Python environment. 
Each task directory contains the following structure:

```bash
benchmarks/
   ├── README.md
   └── <task-name>
            ├── README.md
            ├── main.py
            ├── evaluate.py
            ├── static
            |      └── *.yaml # contains all fixed configurations
            ├── evaluation
            |        └── *.py # contains functions for evaluation
            └──baseline
                    ├── *.py # baseline code as a lower bound performance    
                    └── conf
                          └── *.yaml # config files
```

Some description about each file and directory,

including `static`: contains all fixed configurations

`evaluation`: evaluation functions

that are not allowed to modify.

`baseline/conf` contains configurations for all hyper-parameters
