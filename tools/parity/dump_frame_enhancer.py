"""
Dump ground truth from `facefusion.processors.modules.frame_enhancer` for the C#
FrameEnhancer parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/frame_enhancer` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/frame_enhancer \
        python3 tools/parity/dump_frame_enhancer.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg) and a working internet connection the first
time it runs, since `pre_check()` downloads the real_web_photo_x4 frame enhancer ONNX model
(the smallest tile size of any frame_enhancer model — 64/4/2 — chosen to keep the tile
count, and therefore the fixture set, small) into `.assets/models/` if not already present.

Kept lean per the assignment: rather than tiling the full 1024x1024 source.jpg (which would
produce ~17x17 = 289 tiles at this model's tile size), a small 96x80 crop is used instead —
just enough to exercise `create_tile_frames`/`merge_tile_frames` with a handful of tiles
(2, dumped in full) without inflating the fixture set. `create_tile_frames`/`merge_tile_frames` themselves
already have their own dedicated parity coverage (VisionParityTests); this dumper's job is
the frame_enhancer-specific pieces around them: prepare_tile_frame/forward/normalize_tile_frame
and the final resize+blend.

The highest-value dump is the exact model input tensor (`model_input/tile_0.npy`) handed to
`session.run` for the first tile — if that tensor matches the .NET port's preprocessing
output, any further mismatch is ONNX Runtime's own arithmetic (expect ~0 divergence per
PARITY_HARNESS.md); if it does not match, the bug is in .NET preprocessing
(FrameEnhancer.PrepareTileFrame).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import state_manager
from facefusion.processors.modules.frame_enhancer import core as frame_enhancer
from facefusion.vision import create_tile_frames, read_static_image
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')

MODEL_NAME = 'real_web_photo_x4'


def init_state() -> None:
	for key, value in [
		('download_providers', ['github']),
		('execution_providers', ['cpu']),
		('execution_device_ids', [0]),
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
		('frame_enhancer_model', MODEL_NAME),
		('frame_enhancer_blend', 80),
	]:
		state_manager.init_item(key, value)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	init_state()

	if not frame_enhancer.pre_check():
		raise SystemExit('frame_enhancer.pre_check() failed to fetch the frame_enhancer model.')

	source_frame = read_static_image(SOURCE_IMAGE)
	crop_frame = numpy.ascontiguousarray(source_frame[100:180, 200:296])
	dump('reference/crop_frame', crop_frame)
	dump_scalar('reference/crop_shape', list(crop_frame.shape))

	model_size = frame_enhancer.get_model_options().get('size')
	model_scale = frame_enhancer.get_model_options().get('scale')
	dump_scalar(MODEL_NAME + '/model_size', list(model_size))
	dump_scalar(MODEL_NAME + '/model_scale', model_scale)

	tile_vision_frames, pad_width, pad_height = create_tile_frames(crop_frame, model_size)
	dump_scalar(MODEL_NAME + '/tile_count', len(tile_vision_frames))
	dump_scalar(MODEL_NAME + '/pad_width', pad_width)
	dump_scalar(MODEL_NAME + '/pad_height', pad_height)

	for index, tile_vision_frame in enumerate(tile_vision_frames[:2]):
		dump(MODEL_NAME + f'/tile_{index}/raw', tile_vision_frame)

		prepared_tile_vision_frame = frame_enhancer.prepare_tile_frame(tile_vision_frame)
		dump(MODEL_NAME + f'/tile_{index}/input', prepared_tile_vision_frame)

		forward_output = frame_enhancer.forward(prepared_tile_vision_frame)
		dump(MODEL_NAME + f'/tile_{index}/forward_output', forward_output)

		normalized_tile_vision_frame = frame_enhancer.normalize_tile_frame(forward_output)
		dump(MODEL_NAME + f'/tile_{index}/normalized', normalized_tile_vision_frame)

	# End-to-end enhance_frame(), for an image-level (PSNR/SSIM) sanity check in addition to
	# the stage-by-stage tensors above.
	enhanced_frame = frame_enhancer.enhance_frame(crop_frame)
	dump(MODEL_NAME + '/enhance_frame_output', enhanced_frame)


if __name__ == '__main__':
	main()
