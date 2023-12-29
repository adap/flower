import hydra
from omegaconf import DictConfig, OmegaConf

from hydra.utils import call, instantiate


def function_test(x: int, y: int):
    """A simple function that ads up two integers."""
    print(f"`function_test` received: {x = }, and {y = }")
    result = x + y
    print(f"{result = }")


class MyClass:
    """A simple class."""

    def __init__(self, x):
        self.x = x

    def print_x_squared(self):
        print(f"{self.x**2 = }")


class MyComplexClass:
    """A class with some Hydra magic inside."""

    def __init__(self, my_object: MyClass):
        self.object = my_object

    def instantiate_child(self, value):
        self.object = instantiate(self.object, x=value)


# run main() by passing the `conf/toy.yaml` config file in this directory
@hydra.main(config_path="conf", config_name="toy", version_base=None)
def main(cfg: DictConfig):
    # print config as yaml
    print(OmegaConf.to_yaml(cfg))

    ## THE EASY BITS
    print("--------" * 7)
    # Access elements in your config easily by:
    print(f"{cfg.foo = }")
    print(f"{cfg.bar.baz = }")
    print(f"{cfg.bar.more = }")
    print(
        f"{cfg.bar.more.blabla = }"
    )  # not how when you do access the element directly, it's value gets assigned (not shown as a reference anymore)

    ## NOW I'M INTERESTED !
    print("--------" * 7)
    # Let's run a standard Python function (`function_test` defined above)
    # with the arguments for x and y specified in the config file
    call(cfg.my_func)

    # you can override some arguments at runtime too (which is a bit linked to the
    # topic of the next comment below)
    call(cfg.my_func, x=99)

    # In some settings not all argument will be ready immediately, but you still
    # want to be able to call it from other places in your code once the remaining
    # arguments are defined/updated. You can do this using a Python partial.
    partial_fn = call(
        cfg.my_partial_func
    )  #! If you run this with `_partial_` set to false in the config, it will trigger an error since `y` is not defined

    # with your partial ready you can call the original function by specifying
    # the value of `y` you like.
    partial_fn(y=2023)
    # Aren't partials the second best thing ever?! (after Flower)

    # Similarly to functions, you can create standard Python objects. This time use
    # `instantiate` instead of `call`. The same principle for partials applies here too
    object: MyClass = instantiate(cfg.my_object)
    # then you can call its methods as usual
    object.print_x_squared()  # if you were to type this line yourself you'd notice that there was no autocompletion
    # this is because the class that `object` is an instance of is only know at run time.
    # This makes writing code a bit more inconvenient but it's a small price to pay given
    # how versatile Hydra would make your code base.
    # You could have set the type in the line above by doing: `object: MyClass = instantiate(cfg.my_objcetc)`
    # but, unless all your objects do inherit from a common parent class, this approach might restrict Hydra.

    ## NOW YOU ARE FLYING !!!
    print("--------" * 7)
    # your objects can have other objects inside also defined w/ Hydra
    # by default, `instantiate()` will instantiate everything recursively.
    obj = instantiate(cfg.my_complex_object)
    print(obj.object.x)  # should print 99 unless you changed the config

    # In some situations you might want to have the top-level object defined
    # but not the child objects. Maybe because they can only be instantiated
    # after some other data is available or maybe because you prefer to have
    # more control over the instantiation process. Set `_recursive_` to false
    # in order to prevent recursive instantiation of objects
    obj = instantiate(cfg.my_complex_object_non_recursive)
    print(obj.object)  # you'll see that it retains the config content
    # now let's instantiate setting it's value `x`
    obj.instantiate_child(9999)
    print(
        obj.object.x
    )  # it should print now the value we instantiate the object of type `MyClass` with (i.e. =9999)

    # finally, let's instantiate the PyTorch model set in our config
    # (yes, you can instantiate any class, not just the ones you create)
    model = instantiate(cfg.toy_model)
    # print(model) #! print the model architecture (uncomment)
    num_parameters = sum([p.numel() for p in model.state_dict().values()])
    print(f"{cfg.toy_model} has: {num_parameters} parameters")


if __name__ == "__main__":
    main()
