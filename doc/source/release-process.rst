Release Process
===============

This document describes the current release process. It may or may not change in the future.

Process
-------

The version number of a release is stated in ``pyproject.toml``. To release a new version of Flower, the following things need to happen (in that order):

1. Update the ``changelog.rst`` section header ``Unreleased`` to contain the version number and date for the release you are building. Create a pull request with the change.
2. Tag the release commit with the version number as soon as the PR is merged: ``git tag v0.12.3``, then ``git push --tags``
3. Build the release with ``./dev/build.sh``, then publish it with ``./dev/publish.sh``
4. Create an entry in GitHub releases with the release notes for the previously tagged commit and attach the build artifacts (:code:`.whl` and :code:`.tar.gz`).

After the release
-----------------

Create a pull request which contains the following changes:

1. Increase the minor version in ``pyproject.toml`` by one.
2. Update all files which contain the current version number if necessary.
3. Add a new ``Unreleased`` section in ``changelog.rst``.

Merge the pull request on the same day (i.e., before a new nighly release gets published to PyPI).

Publishing a pre-release
------------------------

Pre-release naming
~~~~~~~~~~~~~~~~~~

PyPI supports pre-releases (alpha, beta, release candiate). Pre-releases MUST use one of the following naming patterns:

- Alpha: ``MAJOR.MINOR.PATCHaN``
- Beta: ``MAJOR.MINOR.PATCHbN``
- Release candiate (RC): ``MAJOR.MINOR.PATCHrcN``

Examples include:

- ``1.0.0a0``
- ``1.0.0b0``
- ``1.0.0rc0``
- ``1.0.0rc1``

This is in line with PEP-440 and the recommendations from the Python Packaging
Authority (PyPA):

- `PEP-440 <https://peps.python.org/pep-0440/>`_
- `PyPA Choosing a versioning scheme <https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#choosing-a-versioning-scheme>`_

Note that the approach defined by PyPA is not compatible with SemVer 2.0.0 spec, for details consult the `Semantic Versioning Specification <https://semver.org/spec/v2.0.0.html#spec-item-11>`_ (specifically item 11 on precedence).

Pre-release classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Should the next pre-release be called alpha, beta, or release candidate?

- RC: feature complete, no known issues (apart from issues that are classified as "won't fix" for the next stable release) - if no issues surface this will become the next stable release
- Beta: feature complete, allowed to have known issues
- Alpha: not feature complete, allowed to have known issues
