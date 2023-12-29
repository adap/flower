# A Complete FL Simulation Pipeline using Flower (w/ better Hydra usage)

The code in this directory is fairly similar to that presented in [`simulation-pytorch example`](https://github.com/adap/flower/tree/main/examples/simulation-pytorch) but extended into a series of [step-by-step video tutorials](https://www.youtube.com/playlist?list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB) on how to Federated Learning simulations using Flower. In Part-I, we made use of a very simple config structure using a single `YAML` file. With the code here presented, we take a dive into more advanced config structures leveraging some of the core functionality of Hydra. You can find more information about Hydra in the [Hydra Documentation](https://hydra.cc/docs/intro/). To the files I have added a fair amount of comments to support and expand upon what was said in the video tutorial.

The content of the code in this directory is roughly divided into two parts:

- `toy.py` and its associated config files (i.e. `conf/toy.yaml` and `conf/toy_model/`) which were designed as a playground to test out some of the functionality of Hydra configs that we want to incorporate into our Flower projects.
- and the rest: which follows the exact same structure as in the code presented in [Part-I](https://github.com/adap/flower/tree/main/examples/flower-simulation-step-by-step-pytorch/Part-I) but that has been _enhanced_ using Hydra.

## Running the Code

You can run the introductory demo code (i.e. `toy.py`) about how to use Hydra as shown below:

```bash
python toy.py # this will run with the default arguments shown in `conf/toy.yaml`

# You can override elements easily
python toy.py foo=456 # will replace foo's default value (123) with 456
python toy.py bar.bazz=48 # will replace bar.bazz's default value (24) with 48
# or change both
python toy.py foo=456 bar.bazz=48

# you can override in this way pretty much anything
python toy.py my_func.x=456 # will replace the x input argument to function `function_test` in toy.py

# to modify the 'defaults' list (see the bottom of `conf/toy.yaml`) the syntax is a bit different
python toy.py toy_model=mobilenetv2 # will replace the default pointing to `resnet18.yaml`
```

Now that you know how to work with Hydra, using more complex config files in your Flower projects (note this can be also applied outside simulation), should feel very intuitive! Let's see a couple of examples on how to launch the simulation:

```bash
python main.py # will launch the simulation with default arguments replicating the exact same setup as in the code for the first part of this tutorial (were we used only a fairly plain .yaml config)

# the default config uses FedAvg, you can override this easily by pointing it instead ot use FedAdam
# internally, Hydra will resolve this command and load `conf/strategy/fedadam.yaml`
python main.py strategy=fedadam

# you can also further change the default config
python main.py strategy=fedadam num_rounds=20 # will use FedAdam and 20 rounds
python main.py strategy=fedadam strategy.tau=0.2 # will use FedAdam and then override its default tau value
```
