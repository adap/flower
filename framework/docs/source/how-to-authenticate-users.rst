:og:description: Configure SuperLink for user authentication and authorization. Private-by-default runs securely with OpenID Connect and OpenFGA.
.. meta::
    :description: Configure SuperLink for user authentication and authorization. Private-by-default runs securely with OpenID Connect and OpenFGA.

Authenticate Users via OpenID Connect
=====================================

.. note::

    OpenID Connect Authentication is a Flower Enterprise feature. See `Flower Enterprise
    <https://flower.ai/enterprise>`_ for details.

In this guide, you'll learn how to configure SuperLink with user-level authentication
and authorization, and how to log in using the ``flwr`` CLI. Once logged in, any users
that are authorized on the SuperLink can run Flower CLI commands that interact with the
SuperLink.

.. important::

    With user authentication and authorization enabled, only users that have submitted
    the ``flwr run`` command can view and interact with their runs. This means that your
    runs are **private by default**, ensuring that only authorized users can access
    them.

Prerequisites
-------------

To enable user authentication and authorization, the SuperLink must be deployed with an
`OpenID Connect (OIDC) <https://openid.net/developers/how-connect-works/>`_ provider and
an `OpenFGA <https://openfga.dev/>`_ server. The OIDC provider is used for user
authentication, while OpenFGA is used for fine-grained access control. This means an
authenticated user can only run ``flwr`` CLI commands on the SuperLink if they have been
granted the necessary permissions by the SuperLink administrator. When enabled, both
user authentication and authorization must be configured on the SuperLink.

Enable User Authentication and Authorization on the SuperLink
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a YAML configuration file with the following content:

.. code-block:: yaml

    authentication:
      auth_type: oidc
      auth_url:           # OIDC provider's authorization_endpoint
      token_url:          # OIDC provider's token_endpoint
      validate_url:       # OIDC provider's userinfo_endpoint
      oidc_client_id:     # OIDC provider Client ID
      oidc_client_secret: # The corresponding Client Secret

    authorization:
      authz_type: openfga
      authz_url:          # The base OpenFGA API URL
      store_id:           # The store ID containing the model store
      model_id:           # The model ID containing the latest authorization model for the SuperLink
      relation:           # The authorized relation between the user and the resource, e.g. `has_access`
      object:             # The object and identifier at which a user has an authorized relation.
                          # The expected format is `object_type:object_identifier`, e.g.:
                          #   service:researchgrid
                          #   │       └─ Identifier of the object
                          #   └─ Object type

Save this file as ``user-auth-config.yaml``. Then pass it to the SuperLink via the
``--user-auth-config`` flag when deploying the SuperLink:

.. code-block:: bash

    ➜ flower-superlink \
        --user-auth-config=user-auth-config.yaml
        <other flags>

.. note::

    To authorize a user, the SuperLink administrator must add the user's OIDC ``sub``
    claim to the OpenFGA store with the appropriate relation.

Login to the SuperLink
----------------------

Once a SuperLink with user authentication and authorization is up and running, a user
can interface with it after installing the ``flwr`` PyPI package via the Flower CLI.
Then, ensure that the ``enable-user-auth`` field is set to ``true`` in the federation
section in the ``pyproject.toml`` of the Flower app you want to run:

.. code-block:: toml

    [tool.flwr.federations]
    default = "my-federation"

    [tool.flwr.federations.my-federation]
    address = "<SUPERLINK-ADDRESS>:9093"   # Address of the SuperLink Control API
    root-certificates = "<PATH/TO/ca.crt>" # TLS certificate set for the SuperLink. Required for self-signed certificates.
    enable-user-auth = true                # Enables the user auth mechanism on the `flwr` CLI side

.. note::

    User authentication and authorization is only supported with TLS connections.

Now, you need to login first before other CLI commands can be executed. Upon executing
``flwr login``, a URL will be returned by the authentication plugin in the SuperLink.
Click on it and authenticate directly against the OIDC provider.

.. code-block:: bash

    flwr login [APP] [FEDERATION]
    Loading project configuration...
    Success
    Creating a new `.gitignore` with `.credentials` entry...
    Please login with your user credentials here: https://account.flower.ai/realms/flower/device?user_code=...
    # [... user follows URL and logs in ... in the meantime the CLI will wait ...]
    ✅ Login successful.

Once the login is successful, the credentials returned by the OIDC provider via the
SuperLink will be saved to the app's directory under
``.flwr/.credentials/<federation-name>.json``. The tokens stored in this file will be
sent transparently with each subsequent ``flwr`` CLI request to the SuperLink, and it
will relay them to OIDC provider to perform the authentication checks.

Run authorized ``flwr`` CLI commands
------------------------------------

With the above steps completed, you can now run ``flwr`` CLI commands against a
SuperLink setup with user authentication and authorization. For example, as an
authorized user, you can run the ``flwr run`` command to start a Flower app:

.. code-block:: bash

    ➜ flwr run
    Loading project configuration...
    Success
    🎊 Successfully built flwrlabs.myawesomeapp.1-0-0.014c8eb3.fab
    🎊 Successfully started run 1859953118041441032

If the user does not have the necessary permissions to run the command, an error will be
returned:

.. code-block:: bash

    ➜ flwr run
    Loading project configuration...
    Success
    ❌ Permission denied.
    User not authorized
