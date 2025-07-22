:og:description: Persist SuperLink state in Flower by saving it to a local directory, ensuring data retention across Docker container restarts.
.. meta::
    :description: Persist SuperLink state in Flower by saving it to a local directory, ensuring data retention across Docker container restarts.

Persist the State of the SuperLink
==================================

By default, the Flower SuperLink keeps its state in-memory. When using the Docker flag
``--rm``, the state is not persisted between container starts.

If you want to persist the state of the SuperLink on your host system, all you need to
do is specify a directory where you want to save the file on your host system and a name
for the database file.

By default, the SuperLink container runs with a non-root user called ``app`` with the
user ID ``49999``. When working with Docker on Linux, it is recommended to create a new
directory and set the user ID to ``49999`` to ensure the mounted directory has the
proper permissions.

If you later want to delete the directory, you can change the user ID back to the
current user ID by running ``sudo chown -R $USER:$(id -gn) state``.

Example
-------

In the example below, we create a new directory called ``state``, change the user ID and
tell Docker via the flag ``--volume`` to mount the local ``state`` directory into the
``/app/state`` directory of the container. Lastly, we use the flag ``--database`` to
specify the name of the database file.

.. code-block:: bash
    :substitutions:

    $ mkdir state
    $ sudo chown -R 49999:49999 state
    $ docker run --rm \
         --volume ./state/:/app/state flwr/superlink:|stable_flwr_version| \
         --database state/state.db \
         <additional-args>

As soon as the SuperLink starts, the file ``state.db`` is created in the ``state``
directory on your host system. If the file already exists, the SuperLink tries to
restore the state from the file. To start the SuperLink with an empty database, ensure
that there is no database called ``state.db`` in the ``state`` directory (``rm
state.db``) before you execute the ``docker run`` command above.
