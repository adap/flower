Installing Virtual Environment
==============================

It is recommended to run your federated setup within a virtual environment.
You can follow the instructions or choose your prefered setup. 

Python Version
--------------

Flower requires `Python 3.6 <https://docs.python.org/3.6/>`_ or above.


Virutualenv with Pyenv/Virtualenv
---------------------------------

One of the recommended virtual environment is `pyenv <https://github.com/pyenv/pyenv>`_/`virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_.Please clone the repositories to download the required libraries.

After your virtualenv installation you can install `Python Version 3.6 <https://docs.python.org/3.6/>`_ or above:

.. code-block:: shell

    pyenv install 3.7.9

Create the virtualenv with:

.. code-block:: shell

    pyenv virtualenv 3.7.9 keras-federated-3.7.9


Activate the virtualenv by running the following command:

.. code-block:: shell

    echo keras-federated-3.7.9 > .python-version


Virtualenv with Poetry
----------------------
