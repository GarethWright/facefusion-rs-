"""
Dump ground truth from the real `facefusion.processors.modules.{face_debugger,frame_colorizer,
background_remover}` for the C# FaceDebugger/FrameColorizer/BackgroundRemover parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/processors2` (or a scratch directory to inspect before
committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/processors2 \
        python3 tools/parity/dump_processors2.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg) and a working internet connection the first time
it runs (downloads the yolo_face detector, ddcolor, deoldify_stable, modnet and u2net_cloth
ONNX models into `.assets/models/` — ddcolor/deoldify_stable are ~900 MB each, so this can take
a few minutes on a slow link).

Kept lean per the assignment ("one image, one or two model families each"):
  - face_debugger: one frame (source.jpg), one detected face (the largest by bounding-box
    area), the default `face_debugger_items` ('face-landmark-5/68 face-mask') with
    `face_mask_types = ['box']` (the face_masker default) so no occluder/parser model is
    needed to exercise the face-mask draw path.
  - frame_colorizer: one family from each `type` ('ddcolor' -> a/b-channel model; 'deoldify' ->
    full-image model, via deoldify_stable), `frame_colorizer_blend = 100` (the default) so the
    final process_frame output equals merge_color_frame's output exactly (no blend-in of the
    original frame to account for separately).
  - background_remover: 'modnet' (a plain single-channel-mask model, the smallest of that
    shape) and 'u2net_cloth' (exercises the `numpy.argmax(..., axis=1)` class-selection
    branch). 'corridor_key' is not dumped — it is the only remaining branch (a second model
    output, its own prepare/merge path) and its two model variants are the largest in the
    model set (1024x1024l/2048x2048); left uncovered and documented as such in the port
    report rather than adding an out-of-budget multi-hundred-MB download for one more branch.
    Fill/despill colours are deliberately non-default (`(0, 255, 0, 255)` fill,
    `(0, 255, 0, 128)` despill) so the dump actually exercises `apply_fill_color`/
    `apply_despill_color`'s blending arithmetic — the CLI's own default fill/despill colour is
    fully transparent (alpha = 0), which would make both a no-op.

The highest-value dump in each case is the exact model input tensor (`.../model_input.npy`)
handed to `session.run` — if it matches the C# port's preprocessing output, any further
mismatch is ONNX Runtime's own arithmetic (expect ~0 divergence there per PARITY_HARNESS.md);
if it does not match, the bug is in .NET preprocessing. For face_debugger (no ONNX model of
its own) the highest-value dump is the rendered frame itself, compared directly rather than
via a model-input tensor.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy

from facefusion import state_manager
from facefusion.face_creator import get_many_faces
from facefusion.processors.modules.background_remover import core as background_remover
from facefusion.processors.modules.face_debugger import core as face_debugger
from facefusion.processors.modules.frame_colorizer import core as frame_colorizer
from facefusion.vision import read_static_image
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')


def init_state() -> None:
	for key, value in [
		('download_providers', ['github']),
		('execution_providers', ['cpu']),
		('execution_device_ids', [0]),
		('face_detector_model', 'yolo_face'),
		('face_detector_size', '640x640'),
		('face_detector_score', 0.5),
		('face_detector_angles', [0]),
		('face_detector_margin', [0, 0, 0, 0]),
		('face_landmarker_model', 'many'),
		('face_landmarker_score', 0.0),
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
		('face_mask_types', ['box']),
		('face_mask_blur', 0.3),
		('face_mask_padding', [0, 0, 0, 0]),
		('face_mask_areas', ['upper-face', 'lower-face', 'mouth']),
		('face_mask_regions', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']),
		('face_occluder_model', 'xseg_1'),
		('face_parser_model', 'bisenet_resnet_34'),
		('face_debugger_items', ['face-landmark-5/68', 'face-mask']),
	]:
		state_manager.init_item(key, value)


# -----------------------------------------------------------------
# face_debugger
# -----------------------------------------------------------------

def dump_face_debugger(source_frame) -> None:
	dump('face_debugger/source_frame', source_frame)

	faces = get_many_faces([source_frame])
	if not faces:
		raise SystemExit('face_debugger: no faces detected on source.jpg')

	areas = [(face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]) for face in faces]
	target_face = faces[int(numpy.argmax(areas))]

	dump('face_debugger/bounding_box', target_face.bounding_box)
	dump('face_debugger/landmark_5', target_face.landmark_set.get('5'))
	dump('face_debugger/landmark_5_68', target_face.landmark_set.get('5/68'))
	dump('face_debugger/landmark_68', target_face.landmark_set.get('68'))
	dump('face_debugger/landmark_68_5', target_face.landmark_set.get('68/5'))
	dump_scalar('face_debugger/angle', int(target_face.angle))
	dump_scalar('face_debugger/origin', target_face.origin)

	rendered_frame = face_debugger.debug_face(target_face, source_frame.copy())
	dump('face_debugger/rendered_frame', rendered_frame)


# -----------------------------------------------------------------
# frame_colorizer
# -----------------------------------------------------------------

def dump_frame_colorizer(source_frame) -> None:
	state_manager.init_item('frame_colorizer_size', '256x256')
	state_manager.init_item('frame_colorizer_blend', 100)

	for model_name in ('ddcolor', 'deoldify_stable'):
		state_manager.init_item('frame_colorizer_model', model_name)

		if not frame_colorizer.pre_check():
			raise SystemExit(f'frame_colorizer.pre_check() failed to fetch {model_name}')

		model_input = frame_colorizer.prepare_temp_frame(source_frame)
		dump(f'frame_colorizer/{model_name}/model_input', model_input)

		final_frame = frame_colorizer.colorize_frame(source_frame)
		dump(f'frame_colorizer/{model_name}/final_frame', final_frame)

		frame_colorizer.clear_inference_pool()


# -----------------------------------------------------------------
# background_remover
# -----------------------------------------------------------------

def dump_background_remover(source_frame) -> None:
	fill_color = (0, 255, 0, 255)
	despill_color = (0, 255, 0, 128)
	state_manager.init_item('background_remover_fill_color', fill_color)
	state_manager.init_item('background_remover_despill_color', despill_color)
	dump_scalar('background_remover/fill_color', list(fill_color))
	dump_scalar('background_remover/despill_color', list(despill_color))

	for model_name in ('modnet', 'u2net_cloth'):
		state_manager.init_item('background_remover_model', model_name)

		if not background_remover.pre_check():
			raise SystemExit(f'background_remover.pre_check() failed to fetch {model_name}')

		model_input = background_remover.prepare_temp_frame(source_frame)
		dump(f'background_remover/{model_name}/model_input', model_input)

		final_frame, final_mask = background_remover.remove_background(source_frame)
		dump(f'background_remover/{model_name}/final_frame', final_frame)
		dump(f'background_remover/{model_name}/final_mask', final_mask)

		background_remover.clear_inference_pool()


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	if not os.path.isfile(SOURCE_IMAGE):
		raise SystemExit(f'Missing example image: {SOURCE_IMAGE} (run tools/parity/fetch_examples.sh)')

	init_state()
	source_frame = read_static_image(SOURCE_IMAGE)

	# Downscale before dumping: source.jpg is 1024x1024, and every fixture below is a full
	# BGR frame (a face-rendered frame, a colorized frame, a background-removed frame) rather
	# than a small tensor, so fixture size scales with frame area. 384x384 is still large
	# enough for yolo_face to reliably find the same largest face (face_debugger only needs
	# one), and none of the three processors' own algorithms depend on the exact source
	# resolution — keeps the corpus lean per PARITY_HARNESS.md without weakening what is being
	# verified (real decode, real detection, real ONNX inference, real cv2 arithmetic, all at
	# a smaller size).
	source_frame = cv2.resize(source_frame, (384, 384))

	dump_face_debugger(source_frame)
	dump_frame_colorizer(source_frame)
	dump_background_remover(source_frame)


if __name__ == '__main__':
	main()
