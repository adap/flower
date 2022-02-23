# Client Selection 

This code shows how to modify a `SimpleClientManager` to select clients based on their `priority` property. To keep things simple, a client's priority will be their own `client_id` (cid). This code builds on top of `simulation_pytorch`.

## Requirements

* Flower 0.18 or later (nightly release, or development version from `main` branch)
* PyTorch 1.7.1 (other versions are likely to work, too)
* Ray 1.4.1

# How to run

Besides the steps described in `examples/simulation_pytorch`, this example:

1. Creates a `Criterion` object that filters clients that have a `priority` property.
2. Modifies a client's `get_properties` to return the client's `priority`. 
3. Defines a `ClientManager` that calls `get_properties` on the server-side `ClientProxy`s and samples
them accordingly.

```bash
$ python main.py
```
