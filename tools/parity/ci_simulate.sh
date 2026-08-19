#!/bin/bash
#
# Runs the .NET test suite the way CI sees it: without the example media and without the
# gitignored models. Both are absent on a GitHub runner, so a suite that is green here but not
# under this script is green only because of files CI does not have.
#
# This exists because that gap has bitten twice. VisionParityTests hardcoded
# /tmp/facefusion-test-examples paths with no skip gate, passed locally for weeks, and failed
# eleven cases on all three runners the first time CI got far enough to run tests at all.
#
# Usage:  tools/parity/ci_simulate.sh
#
# The media and models are moved aside and restored on exit, including on Ctrl-C.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLES=/tmp/facefusion-test-examples
MODELS="$REPO_ROOT/.assets/models"

restore() {
    [ -d "$EXAMPLES-hidden" ] && mv "$EXAMPLES-hidden" "$EXAMPLES"
    [ -d "$MODELS-hidden" ] && mv "$MODELS-hidden" "$MODELS"
    echo "restored: $(ls "$EXAMPLES" 2>/dev/null | wc -l) example files, $(ls "$MODELS" 2>/dev/null | wc -l) model files"
}
trap restore EXIT INT TERM

[ -d "$EXAMPLES" ] && mv "$EXAMPLES" "$EXAMPLES-hidden"
[ -d "$MODELS" ] && mv "$MODELS" "$MODELS-hidden"

cd "$REPO_ROOT"
dotnet test -c Release --nologo "$@"
