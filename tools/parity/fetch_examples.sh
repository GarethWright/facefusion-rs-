#!/usr/bin/env bash
# Fetch the example media the Python test suite downloads in its module fixtures.
#
# The .NET tests look for these in the same location the Python helper uses
# (tests/helper.py: <tempdir>/facefusion-test-examples), so both suites share one copy.
#
# Usage: tools/parity/fetch_examples.sh
set -euo pipefail

EXAMPLES_DIR="${TMPDIR:-/tmp}/facefusion-test-examples"
BASE_URL="https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0"

mkdir -p "$EXAMPLES_DIR"

for file_name in source.jpg source.mp3 target-240p.mp4 target-1080p.mp4; do
    if [ -s "$EXAMPLES_DIR/$file_name" ]; then
        echo "have  $file_name"
    else
        echo "fetch $file_name"
        curl -sSL --fail --retry 3 --max-time 300 -o "$EXAMPLES_DIR/$file_name" "$BASE_URL/$file_name"
    fi
done

echo "examples in $EXAMPLES_DIR"
