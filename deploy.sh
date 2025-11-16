#!/bin/bash
set -e

echo "=== WSI-toolbox Deployment Script ==="
echo ""

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -rf build/ dist/ *.egg-info/
find . -type f -name '*.pyc' -delete
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
echo "✓ Cleanup completed"
echo ""

# Build package
echo "Building package..."
python -m build
echo "✓ Build completed"
echo ""

# Check package integrity
echo "Checking package integrity..."
python -m twine check dist/*
echo "✓ Package check passed"
echo ""

# List built files
echo "Built files:"
ls -lh dist/
echo ""

# Prompt for upload
read -p "Upload to PyPI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Uploading to PyPI..."
    python -m twine upload dist/*
    echo "✓ Deployment completed successfully!"
else
    echo "Upload cancelled."
    echo ""
    echo "To upload manually, run:"
    echo "  python -m twine upload dist/*"
fi
