# Releasing

## Release Naming

A release is named in the format `${A}.${B}.${C}`, where `${A}.${B}` are numbers matching the corresponding MLPerf release, and `${C}` is a patch number. For example: `0.5.0`.

A pre-release is named in the format `${A}.${B}.${C}-rc${D}`, where `${D}` is a release candidate number starting from 1. For example: `0.5.0-rc1`.

## Creating a Release

- Create a release branch from `master`, the branch should be named `${A}.${B}-branch`.
- Before each release cut, edit the `VERSION` file with the exact release name (`${A}.${B}.${C}` or `${A}.${B}.${C}-rc${D}`), and commit the change to the current release branch (NOT the `master` branch). The `VERSION` file is used in package installation so we want to make sure it matches the release version.
- Create a tag from the current release branch, the tag name should be exactly the same as that in the `VERSION` file. This can be done through Github UI.

## Maintaining a Release

- After a release is cut, any changes related to that particular release should be committed to the corresponding release branch.
- Any changes that apply to both future releases and the currently maintained release should be committed to `master` branch and cherry-picked to the current release branch.
