:og:description: Run Flower simulations on AMD Instinct GPUs with ROCm. Configure Ray's AMD GPU discovery via ROCM_PATH and HIP_VISIBLE_DEVICES so the Simulation Runtime can place ClientApps.
.. meta::
    :description: Run Flower simulations on AMD Instinct GPUs with ROCm. Configure Ray's AMD GPU discovery via ROCM_PATH and HIP_VISIBLE_DEVICES so the Simulation Runtime can place ClientApps.

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

#############################
 Run simulations on AMD GPUs
#############################

Flower is framework-agnostic and performs no GPU compute of its own: all device work is
delegated to the ML framework your |clientapp_link|_ uses. On AMD Instinct GPUs,
acceleration therefore comes from a ROCm build of that framework (for example, PyTorch
for ROCm). Flower itself is pure Python, compiles no device code, and runs unmodified --
there is no separate AMD build of ``flwr``.

The Simulation Runtime is a second, independent consumer of the GPU. It uses Ray to
schedule ``ClientApp`` instances, and **Ray discovers AMD GPUs through a different
mechanism than your ML framework does**. The two can disagree: it is common for
``torch.cuda.is_available()`` to return ``True`` while Ray reports zero GPUs. When that
happens, every run that requests a GPU fraction fails before training starts. This guide
covers the AMD-specific setup that keeps the two in agreement.

.. note::

    PyTorch for ROCm exposes AMD GPUs through the ``cuda`` device namespace.
    ``torch.device("cuda")`` and ``torch.cuda.is_available()`` refer to the AMD GPU via
    HIP. No change to your ``ClientApp`` code is required.

The steps below were validated on a single-node AMD Instinct MI300X (gfx942) running
ROCm; other Instinct-class cards behave the same way. Consumer Radeon cards may
additionally need ``HSA_OVERRIDE_GFX_VERSION`` to be set for the ML framework -- a ROCm
concern rather than a Flower one; see AMD's ROCm documentation. Multi-GPU and multi-node
placement follow the same discovery rules described here, but are not covered
separately.

***********************************
 Check that ROCm can reach the GPU
***********************************

This guide assumes a working ROCm installation underneath. ROCm reaches the GPU through
``/dev/kfd`` and ``/dev/dri/renderD*``, which are normally owned by the ``render``
group:

.. code-block:: shell

    ls -l /dev/kfd   # e.g. crw-rw---- 1 root render ...
    id -nG           # must list the owning group

If your account is not in that group, nothing below will help: your ML framework and Ray
will *both* report zero GPUs. Add yourself to the ``render`` and ``video`` groups with
``sudo usermod -aG render,video "$USER"``, then start a new login session. If you run
Flower in a container, pass the devices through instead:

.. code-block:: shell

    docker run --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add render ...

``render`` matters as much as ``video``: ``/dev/kfd`` is owned by it, so a non-root
container that adds only ``video`` still sees zero GPUs. If your distribution owns the
devices with a differently named group, add that one instead -- the ``ls -l`` above
reports it.

The rest of this guide addresses the opposite, more confusing case: ROCm works and your
ML framework sees the GPU, but Ray does not.

******************************************
 Install the ROCm framework before Flower
******************************************

Install your ROCm-enabled ML framework first, following AMD's instructions for that
framework, and only then install Flower:

.. code-block:: shell

    # 1a. ROCm already installed system-wide (e.g. /opt/rocm): install a matching
    #     ROCm build of your framework. Replace "rocm6.4" with the index for your
    #     version; https://pytorch.org/get-started/locally/ reports the right one.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

    # 1b. No system ROCm: install ROCm and the framework together from AMD's
    #     index. This produces the wheel-based layout discussed below. Replace
    #     gfx942 with your architecture, which `rocminfo | grep gfx` reports.
    pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "rocm[libraries,devel,device-gfx942]" "torch[device-gfx942]" \
        "torchvision[device-gfx942]"

    # 2. Flower with the simulation extra, which pulls in Ray
    pip install "flwr[simulation]"

.. important::

    Install the two in separate commands, in this order. Resolving them together can let
    ``pip`` select the default PyPI ``torch`` wheel, which is built for CUDA and will
    never see an AMD GPU. The installation succeeds and only fails later, at the point
    where your ``ClientApp`` tries to use the device.

.. important::

    Installing ROCm wheels into your own environment is not always sufficient. ``flwr
    run`` may build an isolated environment for the run, and a plain ``torch``
    dependency resolves there to the default CUDA wheel -- bypassing the ROCm install
    you just verified. Pin the index in your project's ``pyproject.toml`` so the run
    environment resolves the same wheels:

    .. code-block:: toml

        [[tool.uv.index]]
        name = "rocm"
        url = "https://download.pytorch.org/whl/rocm6.4"
        explicit = true

        [tool.uv.sources]
        torch = { index = "rocm" }
        torchvision = { index = "rocm" }

    Setting ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`` is a quicker alternative: it keeps Ray
    from building a separate environment, so the run uses the interpreter you launched
    from. Pinning the index is the more durable fix, since it travels with the project.

***********************************
 Point Ray at the ROCm SMI library
***********************************

Ray enumerates AMD GPUs by opening ``$ROCM_PATH/lib/librocm_smi64.so``, where
``ROCM_PATH`` defaults to ``/opt/rocm``. Two details matter:

- Ray opens the library by its **unversioned** name. A ROCm tree that ships only a
  versioned ``librocm_smi64.so.<N>`` (for example ``librocm_smi64.so.1``) and no
  ``librocm_smi64.so`` symlink is not sufficient.
- If you installed ROCm from Python wheels rather than system packages, there is usually
  no ``/opt/rocm`` at all, so ``ROCM_PATH`` must be set explicitly.

Confirm the library is where Ray will look for it:

.. code-block:: shell

    test -f "${ROCM_PATH:-/opt/rocm}/lib/librocm_smi64.so" && echo "ROCm SMI found"

If that prints nothing, set ``ROCM_PATH`` to a ROCm root that contains the unversioned
symlink and re-check. When ROCm is installed from wheels, the ``rocm-sdk`` tool reports
the correct root:

.. code-block:: shell

    export ROCM_PATH="$(rocm-sdk path --root)"

``rocm-sdk`` is provided by the ``devel`` extra. If it reports that ``rocm[devel]`` is
not installed, install it -- that package supplies the root you need here.

.. important::

    A wheel-based ROCm installation may expose more than one root. Only the root that
    ships the unversioned ``librocm_smi64.so`` symlink works for Ray: in current AMD
    wheel layouts that is ``_rocm_sdk_devel``, while ``_rocm_sdk_core`` ships only the
    versioned ``librocm_smi64.so.<N>``. Your ML framework locates HIP independently and
    works with either, which is why this misconfiguration shows up as a Ray-only
    failure.

    Published ROCm container recipes commonly set ``ROCM_PATH`` to the *core* root,
    which is correct for the ML framework but leaves Ray unable to enumerate GPUs. If
    you adapt such an image for the Simulation Runtime, override ``ROCM_PATH`` for Ray.

*************************************
 Set ``HIP_VISIBLE_DEVICES`` for Ray
*************************************

Many GPU hosts preset ``ROCR_VISIBLE_DEVICES`` in the login environment. Ray rejects
that variable: when ``ROCR_VISIBLE_DEVICES`` is set and ``HIP_VISIBLE_DEVICES`` is not,
``import ray`` raises ``RuntimeError: Please use HIP_VISIBLE_DEVICES instead of
ROCR_VISIBLE_DEVICES``.

The failure occurs at **import** time, not at ``ray.init()``. Because every Ray worker
re-imports Ray on startup, this also surfaces indirectly as workers that never register.

Move the device selection to the name Ray expects, rather than setting both:

.. code-block:: shell

    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
        export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
        unset ROCR_VISIBLE_DEVICES
    fi

.. important::

    Unset ``ROCR_VISIBLE_DEVICES`` rather than leaving both set. The two masks
    *compose*: ``HIP_VISIBLE_DEVICES`` indexes into the set that
    ``ROCR_VISIBLE_DEVICES`` has already filtered down to. Copying a value such as ``2``
    into both therefore asks HIP for index 2 of a one-device set and exposes no GPU at
    all -- the copy is only safe once ROCr has stopped filtering.

.. important::

    These exports apply to the shell you run them in. A managed SuperLink that is
    already running keeps the environment it started with, and ``flwr run`` reuses that
    process rather than starting a fresh one. After changing ``ROCM_PATH`` or
    ``HIP_VISIBLE_DEVICES``, stop the running SuperLink so the next run picks up the new
    values.

*************************************
 Verify that Ray detects the AMD GPU
*************************************

Checking your ML framework is not enough, since it and Ray discover devices
independently. Check both.

First, the framework:

.. code-block:: shell

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

Then Ray, which is what the Simulation Runtime actually schedules against:

.. code-block:: shell

    python -c "import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()"

The output must contain a non-zero ``GPU`` entry:

.. code-block:: text

    {'CPU': 8.0, 'GPU': 1.0, 'accelerator_type:AMD-Instinct-MI300X-OAM': 1.0, ...}

The ``GPU`` entry is the one that matters. Ray derives ``accelerator_type`` from a
hardcoded table of device IDs, so a card missing from that table is reported without an
``accelerator_type`` label -- it is still counted and still scheduled normally.

Only when ``GPU`` is present and non-zero will a run configured with
``client-resources-num-gpus`` greater than zero be able to place a ``ClientApp``. With
both checks passing, configure and run your simulation exactly as on any other
accelerator -- see :doc:`how-to-run-simulations`:

.. code-block:: shell

    flwr federation simulation-config --num-supernodes 8 --client-resources-num-gpus 0.25
    flwr run . --stream

*****************
 Troubleshooting
*****************

``ActorPool is empty. Stopping Simulation.``
============================================

Ray detected no GPU while your ``ClientApp`` requested a fraction of one. Despite the
message, the process does not always exit: the run can appear to hang after the error is
printed. If that happens, interrupt it and scroll back -- the warning below is emitted
before the hang and is the part that names the cause.

The preceding warning names the mismatch:

.. code-block:: text

    WARNING : Ray reports no GPUs (GPUs=0) but each client requests num_gpus=0.25,
    which no node can satisfy. Either this cluster has no GPU, in which case set
    `num_gpus=0`, or Ray failed to discover one that is present.

On a host where ROCm reaches the GPU, it is the second case: the GPU is present and Ray
did not find it. Lowering ``client-resources-num-gpus`` will not help, because no
non-zero value can be satisfied by a cluster reporting ``GPUs=0``. Work through `Point
Ray at the ROCm SMI library`_ and `Set HIP_VISIBLE_DEVICES for Ray`_, then re-run the
verification above.

When ``/dev/kfd`` is present, Flower follows that warning with a second one naming the
``ROCM_PATH`` requirement directly.

Note that ``torch.cuda.is_available()`` commonly reports ``True`` in exactly this
situation, because your ML framework locates HIP without consulting ``ROCM_PATH``.

Both the ML framework and Ray report zero GPUs
==============================================

When ``torch.cuda.is_available()`` is ``False`` *and* Ray reports no GPU, the problem is
below Flower: ROCm itself cannot reach the device. The usual cause is that your account
lacks access to ``/dev/kfd``, or that a container was started without the device flags.
See `Check that ROCm can reach the GPU`_. The ``ROCM_PATH`` and ``HIP_VISIBLE_DEVICES``
steps in this guide apply only once ROCm works for your ML framework.

``Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES``
==================================================================

See `Set HIP_VISIBLE_DEVICES for Ray`_. Note that this is raised when Ray is imported,
so it can appear either directly in your own process or indirectly as Ray workers that
fail to register.

A run completes but the GPU stays idle
======================================

Restarting the local SuperLink resets the simulation configuration to its defaults,
including ``client-resources-num-gpus`` back to ``0.0``, and the run then completes on
CPU. Apply ``flwr federation simulation-config`` *after* any restart, and confirm the
active value before concluding that a run exercised the GPU:

.. code-block:: shell

    flwr federation ls --federation @none/default
