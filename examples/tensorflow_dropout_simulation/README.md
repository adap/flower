# Simulating Client Dropouts

This example provides a simple solution to simulating dropouts.

## Client-side Logic
Add the following function to your client definition, and call this `check_dropout` function at the start of the `fit` function, as shown in the example.
``` python
def check_dropout(self):
    # Add this whenever you want to simulate dropout
    r = random.random()
    if r < self.dropout_prob:
        raise Exception("Forced Dropout")
```
The author has rewritten the `serde` and `message_handler` classes so that the exception is caught and an `ErrorRes` message will be passed back to the server. That way the client would not crash when it is forced to dropout, and it can be used in another round of training after dropping out in the previous. Do not call `check_dropout` in other functions like `evaluate`, though  one can rewrite the necessary code similarly if they insists.

## Server-side Logic

The `serde` functions on the server side will detect the `ErrorRes` message and throw an exception for that `client_proxy` thread. That way, a failure with name `Forced Dropout` will be shown.

## Usage
To create a client that will dropout under a certain probability, make sure you pases in a `dropout_prob` when declaring an instance. Here is an example:

```python
 # Start Flower client
fl.client.start_numpy_client("localhost:8080", client=CifarClient(dropout_prob=1.0))
```
