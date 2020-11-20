# Release Process

This document describes the current release process. It may or may not change in the future.

## Process

Each week on Monday if there was a commit since the last release create a new release. The version number of this release is in `pyproject.toml`.

When the release is built the following things need to happen:

- [ ] Tag the release commit with the version number
- [ ] Upload the release to PyPI

## After the release

Create a pull request which contains the following changes:

- [ ] Increase the minor version in `pyproject.toml` by one.
- [ ] Update all files which contain the current version number if necessary.
- [ ] Create an entry in the CHANGELOG.md for the new release which documents the changes.
