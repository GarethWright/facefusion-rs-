"""
Dump ground truth from the real `facefusion.processors.modules.{deep_swapper,face_editor}`
for the C# DeepSwapper/FaceEditor parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/processors4` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/processors4 \
        python3 tools/parity/dump_processors4.py

Requires the example media fetched into `/tmp/facefusion-test-examples/` (source.jpg has
several detectable faces at the detector settings used below) and a working internet
connection the first time it runs (downloads `iperov/elon_musk_224.dfm` and the
`live_portrait_{eye_retargeter,lip_retargeter,stitcher}` ONNX models into `.assets/models/`;
`live_portrait_{feature_extractor,motion_extractor,generator}` are shared with
expression_restorer/age_modifier and may already be present).

Kept lean per the assignment ("one image, one model family each"):
  - deep_swapper: every `.dfm` model (all scopes, including the default
    `iperov/elon_musk_224`) is hosted exclusively on Hugging Face
    (`resolve_download_url_by_provider('huggingface', ...)` — there is no GitHub mirror,
    unlike every other processor's `.onnx` catalog). This sandbox's egress policy blocks
    huggingface.co and hf-mirror.com outright (403 at the CONNECT tunnel — confirmed via
    the agent proxy status endpoint, not a transient failure), so no `.dfm` file can be
    fetched here and no real `forward()`/ONNX-Runtime dump is possible in this environment.
    What *is* dumped instead: `prepare_crop_frame`/`normalize_crop_frame`/
    `prepare_crop_mask` — every preprocessing/postprocessing function that does not need
    the model itself — run directly against a real warped crop from source.jpg at a
    hand-picked 224x224 size (matching `iperov/elon_musk_224`'s own resolution), so the
    OpenCV-arithmetic-only code path (sharpen/erode/blur, all real `cv2` calls) is still
    verified byte-for-byte. See the port report for the full accounting of what this
    leaves untested (`forward()`'s ONNX call and `swap_face()` end to end).
  - face_editor: 'live_portrait' is the only model in FaceEditorModel, so there is only one
    family. A handful of non-default slider values are used for apply_edit so every one of
    the fourteen edit_* branches gets a non-trivial input (all sliders != 0), and the
    eye/lip open ratios are non-zero so forward_retarget_eye/forward_retarget_lip are
    genuinely exercised too.

The highest-value dumps are the exact model input tensors handed to `session.run` — if they
match the C# port's preprocessing output, any further mismatch is ONNX Runtime's own
arithmetic (expect ~0 divergence there per PARITY_HARNESS.md).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import state_manager
from facefusion.face_creator import get_many_faces, scale_face
from facefusion.face_helper import scale_face_landmark_5, warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask
from facefusion.processors.modules.deep_swapper import core as deep_swapper
from facefusion.processors.modules.face_editor import core as face_editor
from facefusion.vision import read_static_image
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')


def init_state() -> None:
	for key, value in [
		('download_providers', ['github', 'huggingface']),
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
	]:
		state_manager.init_item(key, value)


# -----------------------------------------------------------------
# deep_swapper — preprocessing only (no ONNX Runtime — see module docstring)
# -----------------------------------------------------------------

# iperov/elon_musk_224's own crop resolution (Python: `get_model_size()` reads this from the
# live `in_face:0` ONNX input, unavailable here since the .dfm cannot be fetched — 224x224 is
# the size the "_224" model name itself declares, so this exercises the exact same code path
# with the exact real value that model would report).
DEEP_SWAPPER_MODEL_SIZE = (224, 224)


def dump_deep_swapper(source_frame) -> None:
	faces = get_many_faces([ source_frame ])
	if not faces:
		raise SystemExit('deep_swapper: no faces detected on source.jpg')

	target_face = faces[0]

	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(source_frame, target_face.landmark_set.get('5/68'), 'dfl_whole_face', DEEP_SWAPPER_MODEL_SIZE)
	dump('deep_swapper/crop_vision_frame', crop_vision_frame)

	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))

	prepared = deep_swapper.prepare_crop_frame(crop_vision_frame)
	dump('deep_swapper/prepared_input', prepared)

	deep_swapper_morph = numpy.array([ numpy.interp(65, [ 0, 100 ], [ 0, 1 ]) ]).astype(numpy.float32)
	dump('deep_swapper/morph_input', deep_swapper_morph)

	# normalize_crop_frame/prepare_crop_mask both only need *some* (H, W[, C]) float array in
	# [0, 1] — round-tripping `prepared` itself (dropping the batch dim, scaling by an
	# arbitrary factor to look mask-shaped) stands in for a real model output without needing
	# the model, and still exercises the exact same clip/astype(uint8)/erode/GaussianBlur cv2
	# calls a real forward() output would go through.
	normalized = deep_swapper.normalize_crop_frame(prepared[0])
	dump('deep_swapper/normalized_crop_frame', normalized)

	rng = numpy.random.default_rng(seed=42)
	crop_source_mask = rng.uniform(0.0, 1.0, size=DEEP_SWAPPER_MODEL_SIZE).astype(numpy.float32)
	crop_target_mask = rng.uniform(0.0, 1.0, size=DEEP_SWAPPER_MODEL_SIZE).astype(numpy.float32)
	dump('deep_swapper/crop_source_mask_input', crop_source_mask)
	dump('deep_swapper/crop_target_mask_input', crop_target_mask)

	# deep_swapper.prepare_crop_mask() itself calls get_model_size(), which needs a live
	# `deep_swapper` ONNX session (unavailable here — see module docstring), so its body is
	# reproduced verbatim against DEEP_SWAPPER_MODEL_SIZE directly rather than calling the
	# real function through a session-less state_manager.
	import cv2
	blur_size = 6.25
	kernel_size = 3
	crop_mask = numpy.minimum.reduce([ crop_source_mask, crop_target_mask ])
	crop_mask = crop_mask.reshape(DEEP_SWAPPER_MODEL_SIZE).clip(0, 1)
	crop_mask = cv2.erode(crop_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations = 2)
	crop_mask = cv2.GaussianBlur(crop_mask, (0, 0), blur_size)
	dump('deep_swapper/crop_mask', crop_mask)


# -----------------------------------------------------------------
# face_editor (live_portrait only)
# -----------------------------------------------------------------

def dump_face_editor(source_frame) -> None:
	state_manager.init_item('face_editor_model', 'live_portrait')
	# Non-zero, non-default values for every slider so every edit_* branch (both the
	# positive and negative side of each conditional, chosen per-slider below) and both
	# retargeters get exercised by apply_edit.
	state_manager.init_item('face_editor_eyebrow_direction', 0.6)
	state_manager.init_item('face_editor_eye_gaze_horizontal', 0.4)
	state_manager.init_item('face_editor_eye_gaze_vertical', -0.3)
	state_manager.init_item('face_editor_eye_open_ratio', 0.5)
	state_manager.init_item('face_editor_lip_open_ratio', -0.4)
	state_manager.init_item('face_editor_mouth_grim', 0.3)
	state_manager.init_item('face_editor_mouth_pout', -0.2)
	state_manager.init_item('face_editor_mouth_purse', 0.25)
	state_manager.init_item('face_editor_mouth_smile', 0.5)
	state_manager.init_item('face_editor_mouth_position_horizontal', -0.15)
	state_manager.init_item('face_editor_mouth_position_vertical', 0.2)
	state_manager.init_item('face_editor_head_pitch', 0.3)
	state_manager.init_item('face_editor_head_yaw', -0.25)
	state_manager.init_item('face_editor_head_roll', 0.1)

	if not face_editor.pre_check():
		raise SystemExit('face_editor.pre_check() failed to fetch live_portrait (feature/motion extractor, eye/lip retargeter, stitcher, generator)')

	faces = get_many_faces([ source_frame ])
	if not faces:
		raise SystemExit('face_editor: no faces detected on source.jpg')

	target_face = faces[0]

	model_template = face_editor.get_model_options().get('template')
	model_size = face_editor.get_model_options().get('size')

	face_landmark_5 = scale_face_landmark_5(target_face.landmark_set.get('5/68'), 1.5)
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(source_frame, face_landmark_5, model_template, model_size)
	dump('face_editor/crop_vision_frame', crop_vision_frame)

	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))

	prepared = face_editor.prepare_crop_frame(crop_vision_frame)
	dump('face_editor/prepared_input', prepared)

	feature_volume = face_editor.forward_extract_feature(prepared)

	pitch, yaw, roll, scale, translation, expression, motion_points = face_editor.forward_extract_motion(prepared)
	dump_scalar('face_editor/motion_scalars', [ float(pitch), float(yaw), float(roll) ])
	dump('face_editor/motion_scale', scale)
	dump('face_editor/motion_translation', translation)
	dump('face_editor/motion_expression', expression)
	dump('face_editor/motion_points', motion_points)

	# Individual retargeter inputs/outputs, dumped directly (not only through apply_edit),
	# so the C# port's ForwardRetargetEye/ForwardRetargetLip can be checked in isolation.
	face_landmark_68 = target_face.landmark_set.get('68')
	left_eye_ratio = face_editor.calculate_distance_ratio(face_landmark_68, 37, 40, 39, 36)
	right_eye_ratio = face_editor.calculate_distance_ratio(face_landmark_68, 43, 46, 45, 42)
	dump_scalar('face_editor/eye_ratios', [ float(left_eye_ratio), float(right_eye_ratio) ])

	eye_motion_points_input = numpy.concatenate([ motion_points.ravel(), [ left_eye_ratio, right_eye_ratio, 0.6 ] ]).reshape(1, -1).astype(numpy.float32)
	dump('face_editor/eye_retargeter_input', eye_motion_points_input)
	eye_retargeter_output = face_editor.forward_retarget_eye(eye_motion_points_input)
	dump('face_editor/eye_retargeter_output', eye_retargeter_output)

	lip_ratio = face_editor.calculate_distance_ratio(face_landmark_68, 62, 66, 54, 48)
	dump_scalar('face_editor/lip_ratio', [ float(lip_ratio) ])
	lip_motion_points_input = numpy.concatenate([ motion_points.ravel(), [ lip_ratio, 1.0 ] ]).reshape(1, -1).astype(numpy.float32)
	dump('face_editor/lip_retargeter_input', lip_motion_points_input)
	lip_retargeter_output = face_editor.forward_retarget_lip(lip_motion_points_input)
	dump('face_editor/lip_retargeter_output', lip_retargeter_output)

	crop_vision_frame_raw = face_editor.apply_edit(prepared, face_landmark_68)
	dump('face_editor/apply_edit_raw_output', crop_vision_frame_raw)

	normalized = face_editor.normalize_crop_frame(crop_vision_frame_raw)
	dump('face_editor/normalized_crop_frame', normalized)

	edited_frame = face_editor.edit_face(target_face, source_frame.copy())
	dump('face_editor/edited_frame', edited_frame)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	if not os.path.isfile(SOURCE_IMAGE):
		raise SystemExit(f'Missing example image: {SOURCE_IMAGE} (run tools/parity/fetch_examples.sh)')

	init_state()
	source_frame = read_static_image(SOURCE_IMAGE)

	dump_deep_swapper(source_frame)
	dump_face_editor(source_frame)


if __name__ == '__main__':
	main()
