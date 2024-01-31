# Image Classification

Describe the attractive story of federated image classification.


> We are changing the way we structure the Flower benchmarks. Some other notes here.



## Installing Dependencies
Project dependencies (such as `torch` and `flwr`) are defined in `requirements.txt`. 
We provide commands to install those dependencies and manage your virtual environment via [pip](https://pip.pypa.io/en/latest/development/), 
but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

Write the command below in your terminal to install the dependencies according to the configuration file `requirements.txt`.

```shell
pip install -r requirements.txt
```

Then, install the static packages using the following command:

```bash
python setup.py install
```

## Running the baseline solution

The `baseline` directory contains the code to run a baseline solution serving as a lower bond.

1. Cloning the flower repository

    ```bash
    git clone https://github.com/adap/flower.git && cd flower
    ```

2. Navigate inside to this directory.

3. Run the baseline with:

   ```bash
   python main.py
   ```
   

## Contributing a new solution (will move to benchmark readme)

Do you have a new solution for this task? Great, we really appreciate your contribution !!

The steps to follow are:

1. Fork the Flower repo and clone it into your machine.
2. Navigate to the `benchmarks/` directory, choose a task.
3. Then, directly modify code in `baseline` directory as well as `main.py`. Also, provide all required packages in `requirement.txt`.
4. Once your code is ready, you just need to create a Pull Request (PR). 
Then, the process to check your solution will begin!


## Evaluate your solution
You can evaluate your trained model with `evaluate.py`.

For centralised evaluation:

```bash
python evaluate.py --model-path=/path/to/model --mode=cen
```

For personalised evaluation:

```bash
python evaluate.py --model-path=/path/to/model --mode=per
```


Further resources:
* [GitHub docs: About forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
* [GitHub docs: Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
* [GitHub docs: Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)



