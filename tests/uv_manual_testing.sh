#!/bin/bash

# A script to test the package with different Python versions manually using uv.
# May be useful for sanity checks before pushing changes to the repository.

# Usage:
# $ bash tests/conda_manual_testing.sh

# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
ENV_ROOT="$PROJECT_DIR/.uv-test-envs"

echo "Running tests from: $SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or not available on PATH."
    exit 1
fi

# Define supported Python versions.
python_versions=("3.10" "3.11" "3.12" "3.13")

# ==============================================================================

cleanup_env() {
    local version="$1"
    rm -rf "$ENV_ROOT/py$version"
}

# Loop over Python versions
for version in "${python_versions[@]}"
do
    env_dir="$ENV_ROOT/py$version"

    echo ""
    echo "=============================================================================="
    echo "Testing VIPRS with Python $version"
    echo "=============================================================================="

    cleanup_env "$version"

    # Create a new uv-managed virtual environment for the Python version.
    uv venv --python "$version" "$env_dir"

    # Check python version:
    "$env_dir/bin/python" --version

    # Install viprs and its test dependencies.
    make -C "$PROJECT_DIR" clean
    uv pip install --python "$env_dir/bin/python" -e "$PROJECT_DIR[test]"

    # List the installed packages:
    uv pip list --python "$env_dir/bin/python"

    # Run pytest
    "$env_dir/bin/python" -m pytest -v "$PROJECT_DIR/tests"

    # Test the CLI scripts:
    PATH="$env_dir/bin:$PATH" bash "$SCRIPT_DIR/test_cli.sh"

    cleanup_env "$version"
done
