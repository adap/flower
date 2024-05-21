Develop in VSCode Dev Containers
================================

When working on the Flower framework we want to ensure that all contributors use the same developer environment to format code or run tests. For this purpose we are using the VSCode Remote Containers extension. What is it? Read the following quote:


  The Visual Studio Code Remote - Containers extension lets you use a Docker container as a fully-featured development environment. It allows you to open any folder inside (or mounted into) a container and take advantage of Visual Studio Code's full feature set. A :code:`devcontainer.json` file in your project tells VS Code how to access (or create) a development container with a well-defined tool and runtime stack. This container can be used to run an application or to separate tools, libraries, or runtimes needed for working with a codebase.

  Workspace files are mounted from the local file system or copied or cloned into the container. Extensions are installed and run inside the container, where they have full access to the tools, platform, and file system. This means that you can seamlessly switch your entire development environment just by connecting to a different container.

Source: `Official VSCode documentation <https://code.visualstudio.com/docs/devcontainers/containers>`_


Getting started
---------------

Configuring and setting up the :code:`Dockerfile` as well the configuration for the devcontainer can be a bit more involved. The good thing is you don't have to do it. Usually it should be enough to install `Docker <https://docs.docker.com/engine/install/>`_ on your system and ensure its available on your command line. Additionally, install the `VSCode Containers Extension <vscode:extension/ms-vscode-remote.remote-containers>`_.

Now you should be good to go. When starting VSCode, it will ask you to run in the container environment and - if you confirm - automatically build the container and use it. To manually instruct VSCode to use the devcontainer, you can, after installing the extension, click the green area in the bottom left corner of your VSCode window and select the option *(Re)Open Folder in Container*.

In some cases your setup might be more involved. For those cases consult the following sources:

* `Developing inside a Container <https://code.visualstudio.com/docs/devcontainers/containers#_system-requirements>`_
* `Remote development in Containers <https://code.visualstudio.com/docs/devcontainers/tutorial>`_
