:og:description: Configure SuperExec authentication in Flower using signed metadata and a SuperExec auth YAML file.
.. meta::
    :description: Configure SuperExec authentication in Flower using signed metadata and a SuperExec auth YAML file.

########################
 Authenticate SuperExecs
########################

This guide explains how SuperExec authentication works for AppIO RPCs and how to create
``superexec_auth_config.yaml``.

At a high level, authentication is signature-based:

- SuperExec signs request metadata with its private key.
- SuperLink verifies the signature, timestamp, request method, and caller key
  authorization.
- Authorization is controlled by ``public_keys`` entries in
  ``superexec_auth_config.yaml``.

*************************
 Runtime behavior summary
*************************

When SuperExec auth is enabled, SuperLink enforces the following on protected AppIO
calls:

- metadata headers must be present (public key, signature, timestamp)
- public key must be authorized for the plugin type
- signature must match ``"{timestamp}\\n{grpc_method}"``
- timestamp must be fresh (within tolerance window)

Current AppIO behavior:

- ``ListAppsToLaunch``: SuperExec signed metadata required (when enabled)
- ``RequestToken``: SuperExec signed metadata required (when enabled)
- ``GetRun``: exactly one auth mechanism when enabled:
  - valid run token, or
  - valid SuperExec signed metadata

If you do not pass ``--superexec-auth-config`` to ``flower-superlink``, SuperExec auth
is disabled.

*******************************
 ``superexec_auth_config.yaml``
*******************************

Supported fields:

- ``enabled`` (bool, optional)
  - default: ``true`` when the file is provided
  - set to ``false`` to keep SuperExec auth configured but disabled
- ``timestamp_tolerance_sec`` (int, optional)
  - default: ``300``
  - maximum age allowed for signed timestamps (clock drift allowance is also applied)
- ``public_keys`` (list, optional but required when ``enabled: true``)
  - list of authorized SuperExec public keys
  - each entry is either:
    - a string (public key, authorized for all supported plugin types), or
    - a mapping with:
      - ``public_key`` (string, required)
      - ``allowed_plugins`` (string or list of strings, optional)

Allowed plugin labels in ``allowed_plugins``:

- ``serverapp``
- ``simulation``

Public key requirements:

- key must be an EC public key on a NIST curve
- SSH public key and PEM public key formats are accepted

*************************************
 Key scope: all plugins vs per-plugin
*************************************

You can authorize keys in two ways:

- One key for all plugin types
  - easier key management
  - less separation between ``serverapp`` and ``simulation`` identities
- Different keys per plugin type
  - stronger separation of identities and permissions
  - more operational overhead (more keys to manage/rotate)

If you are starting simple, one shared key is usually acceptable. If you need stronger
control boundaries, use per-plugin keys.

**************
 YAML examples
**************

Shared key for both ``serverapp`` and ``simulation``:

.. code-block:: yaml

    enabled: true
    timestamp_tolerance_sec: 300
    public_keys:
      - "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAA..."

Per-plugin key scoping:

.. code-block:: yaml

    enabled: true
    timestamp_tolerance_sec: 300
    public_keys:
      - public_key: "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAA...serverapp"
        allowed_plugins: serverapp
      - public_key: |
          -----BEGIN PUBLIC KEY-----
          MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE...
          -----END PUBLIC KEY-----
        allowed_plugins:
          - simulation

Disabled (explicit):

.. code-block:: yaml

    enabled: false
    timestamp_tolerance_sec: 300
    public_keys: []

**************
 Launching CLI
**************

Subprocess isolation (SuperLink spawns SuperExec):

.. code-block:: bash

    flower-superlink \
      --isolation subprocess \
      --superexec-auth-config /path/to/superexec_auth_config.yaml \
      --auth-superexec-private-key /path/to/superexec_private_key

Notes:

- with ``--isolation=subprocess`` and ``enabled: true``, the private key argument above
  is required
- SuperLink forwards that private key path to the spawned ``flower-superexec``

Process isolation (you run SuperExec separately):

.. code-block:: bash

    flower-superlink \
      --isolation process \
      --superexec-auth-config /path/to/superexec_auth_config.yaml

.. code-block:: bash

    flower-superexec \
      --plugin-type serverapp \
      --appio-api-address 127.0.0.1:9091 \
      --auth-superexec-private-key /path/to/superexec_private_key

Important:

- in ``--isolation=process``, passing ``--auth-superexec-private-key`` to
  ``flower-superlink`` does not configure an external SuperExec process
- pass the private key directly to each external ``flower-superexec``

**********************
 Common configuration errors
**********************

- ``enabled: true`` but no ``public_keys``: SuperLink exits with invalid configuration
- invalid key format or unsupported curve: SuperLink rejects the key
- unknown ``allowed_plugins`` value: SuperLink rejects the config

