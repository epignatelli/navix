# get current version
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
echo "Script dir is: $SCRIPT_DIR"
VERSION=$SCRIPT_DIR/../VERSION
echo "Current version is: $(cat $VERSION)"

# cd to repo dir
REPO_DIR="$(cd "$(dirname -- "$1")" >/dev/null; pwd -P)/$(basename -- "$1")"
echo "Repo dir is: $REPO_DIR"
cd $REPO_DIR

# create tag
git tag -a $(cat $VERSION) -m "Release $(cat $VERSION)"
git push origin $(cat $VERSION)

# create release
gh release create $VERSION

# trigger CD
gh workflow run cd.yml