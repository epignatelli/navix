#!/bin/bash

# get current script directory
SCRIPT="$(readlink -f "$0")"
SCRIPT_DIR=$(dirname "$SCRIPT")
echo "Script dir is: $SCRIPT_DIR"

# get version file
VERSION_FILE="$SCRIPT_DIR/../navix/_version.py"
VERSION_CONTENT="$(cat "$VERSION_FILE")"
echo "Version file found at: $VERSION_FILE and contains:"
echo "$VERSION_CONTENT"

# extract version
VERSION=$(cat navix/_version.py | grep "__version__ = " |  cut -d'=' -f2 | sed 's,\",,g' | sed "s,',,g" | sed 's, ,,g')
echo "Current version is:"
echo "$VERSION"

# cd to repo dir
REPO_DIR="$(cd "$(dirname -- "$1")" >/dev/null; pwd -P)/$(basename -- "$1")"
echo "Repo dir is: $REPO_DIR"
cd $REPO_DIR

# create tag
git tag -a $VERSION -m "Release $VERSION"
git push --tags

# create release
gh release create $VERSION

# trigger CD
gh workflow run CD -r main
