# Release Process

This document describes the current release process. It may or may not change in the future.

## Process

Each week on Friday if there was a commit since the last release create a new release. The version number of this release is in the `pyproject.toml`.

When the release is build the following things need to happen

- [ ] Tag the released commit with the version number
- [ ] Upload the release to PyPi

### After the release

Create a pull request which contains the following changes

- [ ] Increases the minor version in the `pyproject.toml`by one.
- [ ] Adjust all files which contain the current version number if necessary.
- [ ] Create an entry in the CHANGELOG.md for the new release which documents the changes.
