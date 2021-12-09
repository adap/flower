# client_selection 

This code shows how to modify a `SimpleClientManager` to select clients based on their `priority` proterty. To keep things simple, a client's priority will be their own `client_id` (cid).
This code builds on top of `simulation_pytorch`.

## Requirements

*    Flower nightly release (or development version from `main` branch)
*    PyTorch 1.7.1 (but most likely will work with older versions)
*    Ray 1.4.1

# How to run

Besides the steps decrived in `examples/simulation_pytorch`, this example:

1. Creates a `Criterion` object that is filter clients that have a `priority`.
2. Modifies a client's `get_parameters` to return the client's `priority`. 
3. Defines a `ClientManager` that calls `get_parameters` from `ClientProxy`s and samples
them accordingly.

```bash
$ python main.py
```
