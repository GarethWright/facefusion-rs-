#!/usr/bin/env bash
# Prepare the example media the test suites use.
#
# Mirrors the Python suite's module fixtures: tests/test_ffmpeg.py downloads three files
# and then DERIVES a set of variants with ffmpeg. The .NET tests assert against the same
# derived files, so they must be produced identically here.
#
# Both suites read from <tempdir>/facefusion-test-examples (tests/helper.py and
# tests/FaceFusion.UnitTests/TestHelper.cs resolve to the same path), so one copy serves
# both.
#
# Usage: tools/parity/fetch_examples.sh
set -euo pipefail

EXAMPLES_DIR="${TMPDIR:-/tmp}/facefusion-test-examples"
BASE_URL="https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0"

mkdir -p "$EXAMPLES_DIR"

# --- downloaded ----------------------------------------------------------------------
# target-1080p.mp4 is not used by the Python fixtures but is handy for parity dumps.
for file_name in source.jpg source.mp3 target-240p.mp4 target-1080p.mp4; do
    if [ -s "$EXAMPLES_DIR/$file_name" ]; then
        echo "have    $file_name"
    else
        echo "fetch   $file_name"
        curl -sSL --fail --retry 3 --max-time 300 -o "$EXAMPLES_DIR/$file_name" "$BASE_URL/$file_name"
    fi
done

FFMPEG_FILTERS=""
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_FILTERS="$(ffmpeg -hide_banner -filters 2>/dev/null || true)"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found - skipping derived fixtures; ffmpeg-dependent tests will skip" >&2
    exit 0
fi

# --- derived (see tests/test_ffmpeg.py::before_all) ------------------------------------
derive() {
    local output="$1"; shift
    if [ -s "$EXAMPLES_DIR/$output" ]; then
        echo "have    $output"
    else
        echo "derive  $output"
        ffmpeg -loglevel error -y "$@" "$EXAMPLES_DIR/$output"
    fi
}

derive source.wav -i "$EXAMPLES_DIR/source.mp3"

for video_fps in 25 30 60; do
    derive "target-240p-${video_fps}fps.mp4" -i "$EXAMPLES_DIR/target-240p.mp4" -vf "fps=${video_fps}"
done

# The Python fixture uses `scale=out_transfer=smpte2084`, but the `out_transfer` option
# only exists on ffmpeg 7+; on ffmpeg 6.x (what Ubuntu 24.04 ships) that fails, and the
# Python fixture would fail here too. `zscale=transfer=smpte2084` produces an equivalent
# PQ-transfer clip, which is all the fixture is for - exercising restrict_color_transfer.
if [ -s "$EXAMPLES_DIR/target-240p-smpte2084.mp4" ]; then
    echo "have    target-240p-smpte2084.mp4"
elif [ -n "$FFMPEG_FILTERS" ] && case "$FFMPEG_FILTERS" in *" zscale "*) true ;; *) false ;; esac; then
    echo "derive  target-240p-smpte2084.mp4 (zscale)"
    ffmpeg -loglevel error -y -i "$EXAMPLES_DIR/target-240p.mp4" -vf 'zscale=transfer=smpte2084' "$EXAMPLES_DIR/target-240p-smpte2084.mp4"
elif ffmpeg -loglevel error -y -i "$EXAMPLES_DIR/target-240p.mp4" -vf 'scale=out_transfer=smpte2084' "$EXAMPLES_DIR/target-240p-smpte2084.mp4" 2>/dev/null; then
    echo "derive  target-240p-smpte2084.mp4 (scale)"
else
    echo "skip    target-240p-smpte2084.mp4 - needs ffmpeg 7+ scale=out_transfer or zscale" >&2
fi

for output_video_format in avi m4v mkv mov mp4 webm wmv; do
    derive "target-240p-16khz.${output_video_format}" \
        -i "$EXAMPLES_DIR/source.mp3" -i "$EXAMPLES_DIR/target-240p.mp4" -ar 16000
done

derive target-240p-48khz.mp4 -i "$EXAMPLES_DIR/source.mp3" -i "$EXAMPLES_DIR/target-240p.mp4" -ar 48000

echo "examples in $EXAMPLES_DIR"
