Virtual Env Installation
========================

It is recommended to run your Python setup within a virtual environment.
This guide shows three different examples how to create a virtual environment with pyenv virtualenv, poetry, or Anaconda.
You can follow the instructions or choose your preferred setup. 

Python Version
--------------

Flower requires `Python 3.7 <https://docs.python.org/3.7/>`_ or above, we recommend `Python 3.8 <https://docs.python.org/3.8/>`_.

Virutualenv with Pyenv/Virtualenv
---------------------------------

One of the recommended virtual environment is `pyenv <https://github.com/pyenv/pyenv>`_/`virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_. Please see `Flower examples <https://github.com/adap/flower/tree/main/examples/>`_ for details.

Once Pyenv is set up, you can use it to install `Python Version 3.7 <https://docs.python.org/3.7/>`_ or above:

.. code-block:: shell

    pyenv install 3.7.13

Create the virtualenv with:

.. code-block:: shell

    pyenv virtualenv 3.7.13 flower-3.7.13


Activate the virtualenv by running the following command:

.. code-block:: shell

    echo flower-3.7.13 > .python-version


Virtualenv with Poetry
----------------------

The Flower examples are based on `Poetry <https://python-poetry.org/docs/>`_ to manage dependencies. After installing Poetry you simply create a virtual environment with:

.. code-block:: shell

    poetry shell

If you open a new terminal you can activate the previously created virtual environment with the following command:

.. code-block:: shell

    source $(poetry env info --path)/bin/activate


Virtualenv with Anaconda
------------------------

If you prefer to use Anaconda for your virtual environment then install and setup the `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_  package. After setting it up you can create a virtual environment with:

.. code-block:: shell

    conda create -n flower-3.7.13 python=3.7.13

and activate the virtual environment with:

.. code-block:: shell

    conda activate flower-3.7.13


And then?
---------

As soon as you created your virtual environment you clone one of the `Flower examples <https://github.com/adap/flower/tree/main/examples/>`_. 
