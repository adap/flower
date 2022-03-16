Release Process
===============

This document describes the current release process. It may or may not change in the future.

Process
-------

Every four weeks on Monday if there was a commit since the last release create a new release. The version number of this release is in `pyproject.toml`.

To make a release the following things need to happen:

1. Update the `changelog.rst` section header :code:`Unreleased` to contain the version number and date for the release you are building. Create a pull request with the change.
3. If it's a major or minor release (but not a patch release), create and checkout a release branch named :code:`git checkout -b release/1.2`
2. Tag the release commit with the version number as soon as the PR is merged: :code:`git tag v1.2.3`, then :code:`git push --tags`
3. Build the release with `./dev/build.sh`, then publish it with `./dev/publish.sh`
4. Create an entry in GitHub releases with the release notes for the previously tagged commit and attach the build artifacts (:code:`.whl` and :code:`.tar.gz`).

After the release
-----------------

Create a pull request which contains the following changes:

1. Increase the minor version in `pyproject.toml` by one.
2. Update all files which contain the current version number if necessary.
3. Add a new :code:`Unreleased` section in `changelog.rst`.

Merge the pull request on the same day (i.e., before a new nighly release gets published to PyPI).
