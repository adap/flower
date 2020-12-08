Virtual Env Installation
================

It is recommended to run your federated setup within a virtual environment.
We give you three different examples how to create the virtual environment with pyenv virtualenv, poetry, or anaconda. 
You can follow the instructions or choose your prefered setup. 

Python Version
--------------

Flower requires `Python 3.6 <https://docs.python.org/3.6/>`_ or above.


Virutualenv with Pyenv/Virtualenv
---------------------------------

One of the recommended virtual environment is `pyenv <https://github.com/pyenv/pyenv>`_/`virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_. Please clone the repositories to download the required libraries.

After your virtualenv installation you can install `Python Version 3.6 <https://docs.python.org/3.6/>`_ or above:

.. code-block:: shell

    pyenv install 3.7.9

Create the virtualenv with:

.. code-block:: shell

    pyenv virtualenv 3.7.9 ml-federated-3.7.9


Activate the virtualenv by running the following command:

.. code-block:: shell

    echo ml-federated-3.7.9 > .python-version


Virtualenv with Poetry
----------------------

The flower examples are based on `poetry <https://python-poetry.org/docs/>`_ to manage all dependency and ensure the right version of them. After installing poetry you simply create a virtual environment with:

.. code-block:: shell

    poetry shell

If you open a new terminal you can activate the previously created virtual environment with the following command:

.. code-block:: shell

    source $(poetry env info --path)/bin/activate


Virtualenv with Anaconda
------------------------

If you prefer to use anaconda for your virtual environment then install and setup the `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_  package. After setting it up you can create a virtual environment with:

.. code-block:: shell

    conda create -n ml-federated-3.7.9 python=3.7.9

and activate the virtual environment with:

.. code-block:: shell

    conda activate ml-federated-3.7.9


And then?
---------

As soon as you created your virtual environment you can clone the repository and use poetry to install all required packages as shown in the `Flower examples <https://github.com/adap/flower/tree/main/examples/>`_. 