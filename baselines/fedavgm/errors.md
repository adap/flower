# Errors while working with the baseline template

## Issue about poetry x Python 3.10.12 x ray==1.11.1

- Solution: downgrade to Python 3.9.16

```terminal
$ poetry install
...
...

  • Installing ray (1.11.1): Failed

  RuntimeError

  Unable to find installation candidates for ray (1.11.1)

  at ~/.local/share/pypoetry/venv/lib/python3.8/site-packages/poetry/installation/chooser.py:73 in choose_for
       69│ 
       70│             links.append(link)
       71│ 
       72│         if not links:
    →  73│             raise RuntimeError(f"Unable to find installation candidates for {package}")
       74│ 
       75│         # Get the best link
       76│         chosen = max(links, key=lambda link: self._sort_key(package, link))
       77│ 

  • Installing ruff (0.0.272)
  • Installing types-requests (2.27.7)
```

## Issue about poetry x Tensorflow

- Solution: downgrade to tensorflow==2.10 :warning:

```terminal
$ poetry add tensorflow
...
...

The current project's Python requirement (>=3.8.15,<3.12.0) is not compatible with some of the required packages Python requirement:
  - tensorflow requires Python >=3.9, so it will not be satisfied for Python >=3.8.15,<3.9

Because no versions of tensorflow match >2.13.0,<2.14.0rc0 || >2.14.0rc0,<3.0.0
 and tensorflow (2.14.0rc0) requires Python >=3.9, tensorflow is forbidden.
And because tensorflow (2.13.0) depends on tensorboard (>=2.13,<2.14)
 and no versions of tensorboard match >2.13,<2.14, tensorflow (>=2.13.0,<3.0.0) requires tensorboard (2.13).
And because tensorboard (2.13.0) depends on grpcio (>=1.48.2)
 and ray (1.11.1) depends on grpcio (>=1.28.1,<=1.43.0), tensorflow (>=2.13.0,<3.0.0) is incompatible with ray (1.11.1).
So, because fedavgm depends on both ray (1.11.1) and tensorflow (^2.13.0), version solving failed.

  • Check your dependencies Python requirement: The Python requirement can be specified via the `python` or `markers` properties
    
    For tensorflow, a possible solution would be to set the `python` property to ">=3.9,<3.12.0"
```
