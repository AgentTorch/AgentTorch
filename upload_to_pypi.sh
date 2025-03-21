#!/bin/bash

# Exit on error
set -e

# Build the package
echo "Building package..."
python -m build

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "Upload complete!" 