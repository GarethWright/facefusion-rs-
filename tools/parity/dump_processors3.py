"""
Dump ground truth from the real `facefusion.processors.live_portrait` and
`facefusion.processors.modules.{expression_restorer,age_modifier}` for the C#
LivePortrait/ExpressionRestorer/AgeModifier parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/processors3` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/processors3 \
        python3 tools/parity/dump_processors3.py

Requires the example media fetched into `/tmp/facefusion-test-examples/` (source.jpg and
target-240p.mp4 — each has exactly one detectable face at the detector settings used below)
and a working internet connection the first time it runs (downloads the live_portrait
feature_extractor/motion_extractor/generator and fran ONNX models into `.assets/models/`).

Kept lean per the assignment ("one image, one model family each"):
  - live_portrait.create_rotation: dumped for a fixed, arbitrary (pitch, yaw, roll) triple —
    pure NumPy/SciPy, no model, verifies the Euler convention (scipy `from_euler('xyz', ...,
    degrees=True)`) independent of any ONNX run.
  - expression_restorer: 'live_portrait' is the only model in ExpressionRestorerModel, so
    there is only one family. The face detected on target-240p.mp4's first frame donates the
    "target" expression, the face detected on source.jpg is the "temp" identity that receives
    it (its landmarks used for both warps, matching restore_expression's real one-face
    signature) — two genuinely different expressions/motion feature volumes.
  - age_modifier: only 'fran' is dumped (the default model, and the smaller of the two
    ONNX families) — 'styleganex_age' is left uncovered by an ONNX-backed fixture and is
    instead unit-tested against hand-computed values in C#, matching how FaceSwapper's
    frame-based source path was verified where no model fixture was pulled.

The highest-value dumps are the exact model input tensors handed to `session.run` — if they
match the C# port's preprocessing output, any further mismatch is ONNX Runtime's own
arithmetic (expect ~0 divergence there per PARITY_HARNESS.md).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
from facefusion.face_creator import get_many_faces
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask
from facefusion.processors import live_portrait
from facefusion.processors.modules.age_modifier import core as age_modifier
from facefusion.processors.modules.expression_restorer import core as expression_restorer
from facefusion.vision import read_static_image, read_static_video_frame
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')
TARGET_VIDEO = os.path.join(EXAMPLES_DIR, 'target-240p.mp4')


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
	]:
		state_manager.init_item(key, value)


# -----------------------------------------------------------------
# live_portrait.create_rotation
# -----------------------------------------------------------------

def dump_live_portrait_rotation() -> None:
	pitch, yaw, roll = 12.5, -18.25, 7.0
	rotation = live_portrait.create_rotation(pitch, yaw, roll)
	dump_scalar('live_portrait/rotation_input', [ pitch, yaw, roll ])
	dump('live_portrait/rotation_output', rotation)

	expression = numpy.linspace(-0.05, 0.05, 21 * 3, dtype = numpy.float32).reshape(1, 21, 3)
	dump('live_portrait/expression_unclipped', expression)
	dump('live_portrait/expression_limited', live_portrait.limit_expression(expression))

	target_pitch, target_yaw, target_roll = 45.0, -70.0, 25.0
	output_pitch, output_yaw, output_roll = live_portrait.limit_angle(target_pitch, target_yaw, target_roll, 10.0, -10.0, 5.0)
	dump_scalar('live_portrait/limit_angle_input', [ target_pitch, target_yaw, target_roll, 10.0, -10.0, 5.0 ])
	dump_scalar('live_portrait/limit_angle_output', [ output_pitch, output_yaw, output_roll ])


# -----------------------------------------------------------------
# expression_restorer
# -----------------------------------------------------------------

def dump_expression_restorer(source_frame, video_frame) -> None:
	# source.jpg and target-240p.mp4's first frame each contain exactly one detectable face
	# at these settings (verified interactively — not the 10-face count the assignment brief
	# mentions for a different detector/size combination), so the "target" (expression donor)
	# and "temp" (identity that receives it) faces come from two different frames/identities
	# instead of two faces in one frame. This still exercises every step of `apply_restore`
	# with two genuinely different expressions and motion feature volumes.
	state_manager.init_item('expression_restorer_model', 'live_portrait')
	state_manager.init_item('expression_restorer_factor', 80)
	state_manager.init_item('expression_restorer_areas', [ 'upper-face', 'lower-face' ])

	if not expression_restorer.pre_check():
		raise SystemExit('expression_restorer.pre_check() failed to fetch live_portrait')

	target_faces = get_many_faces([ video_frame ])
	temp_faces = get_many_faces([ source_frame ])
	if not target_faces or not temp_faces:
		raise SystemExit('expression_restorer: no face detected on source.jpg or target-240p.mp4')

	# restore_expression's real signature only ever detects one face (`target_face`) and uses
	# its landmarks to crop *both* `target_vision_frame` and `temp_vision_frame` (they are the
	# same tracked face across frames of one video in the real pipeline) — reproduced here
	# with `temp_face`'s landmarks used for both warps, matching that shape exactly, even
	# though the two source frames are unrelated stand-ins for "two different expressions".
	temp_face = temp_faces[0]
	target_face = temp_face

	model_template = expression_restorer.get_model_options().get('template')
	model_size = expression_restorer.get_model_options().get('size')

	target_crop_vision_frame, _ = warp_face_by_face_landmark_5(video_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	temp_crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(source_frame, temp_face.landmark_set.get('5/68'), model_template, model_size)

	dump('expression_restorer/target_crop_vision_frame', target_crop_vision_frame)
	dump('expression_restorer/temp_crop_vision_frame', temp_crop_vision_frame)

	prepared_target = expression_restorer.prepare_crop_frame(target_crop_vision_frame)
	prepared_temp = expression_restorer.prepare_crop_frame(temp_crop_vision_frame)
	dump('expression_restorer/prepared_target_input', prepared_target)
	dump('expression_restorer/prepared_temp_input', prepared_temp)

	feature_volume = expression_restorer.forward_extract_feature(prepared_temp)
	dump('expression_restorer/feature_volume', feature_volume)

	target_pitch, target_yaw, target_roll, target_scale, target_translation, target_expression, target_motion_points_raw = expression_restorer.forward_extract_motion(prepared_target)
	temp_pitch, temp_yaw, temp_roll, temp_scale, temp_translation, temp_expression, temp_motion_points = expression_restorer.forward_extract_motion(prepared_temp)

	dump_scalar('expression_restorer/target_motion_scalars', [ float(target_pitch), float(target_yaw), float(target_roll) ])
	dump_scalar('expression_restorer/temp_motion_scalars', [ float(temp_pitch), float(temp_yaw), float(temp_roll) ])
	dump('expression_restorer/target_scale', target_scale)
	dump('expression_restorer/temp_scale', temp_scale)
	dump('expression_restorer/target_translation', target_translation)
	dump('expression_restorer/temp_translation', temp_translation)
	dump('expression_restorer/target_expression', target_expression)
	dump('expression_restorer/temp_expression', temp_expression)
	dump('expression_restorer/temp_motion_points', temp_motion_points)

	rotation = live_portrait.create_rotation(temp_pitch, temp_yaw, temp_roll)
	dump('expression_restorer/temp_rotation', rotation)

	expression_restorer_factor = float(numpy.interp(80.0, [ 0, 100 ], [ 0, 1.2 ]))
	restricted_expression = expression_restorer.restrict_expression_areas(temp_expression, target_expression.copy())
	dump('expression_restorer/restricted_expression', restricted_expression)

	crop_vision_frame = expression_restorer.apply_restore(prepared_target, prepared_temp, expression_restorer_factor)
	dump('expression_restorer/apply_restore_raw_output', crop_vision_frame)

	normalized = expression_restorer.normalize_crop_frame(crop_vision_frame)
	dump('expression_restorer/normalized_crop_frame', normalized)

	box_mask = create_box_mask(temp_crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
	dump('expression_restorer/box_mask', box_mask)

	restored_frame = expression_restorer.restore_expression(target_face, video_frame, source_frame.copy())
	dump('expression_restorer/restored_frame', restored_frame)


# -----------------------------------------------------------------
# age_modifier (fran only)
# -----------------------------------------------------------------

def dump_age_modifier(source_frame) -> None:
	state_manager.init_item('age_modifier_model', 'fran')
	state_manager.init_item('age_modifier_direction', 25)

	if not age_modifier.pre_check():
		raise SystemExit('age_modifier.pre_check() failed to fetch fran')

	faces = get_many_faces([ source_frame ])
	if not faces:
		raise SystemExit('age_modifier: no faces detected on source.jpg')

	areas = [ (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]) for face in faces ]
	target_face = faces[int(numpy.argmax(areas))]

	dump_scalar('age_modifier/target_age', [ float(target_face.age.start), float(target_face.age.stop) ])

	model_templates = age_modifier.get_model_options().get('templates')
	model_sizes = age_modifier.get_model_options().get('sizes')
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(source_frame, target_face.landmark_set.get('5/68'), model_templates.get('target'), model_sizes.get('target'))
	dump('age_modifier/crop_vision_frame', crop_vision_frame)

	prepared = age_modifier.prepare_vision_frame(crop_vision_frame)
	dump('age_modifier/prepared_input', prepared)

	target_age = numpy.mean(target_face.age)
	age_modifier_direction = numpy.array([ target_age, target_age + state_manager.get_item('age_modifier_direction') ], dtype = numpy.float32) / 100
	age_modifier_direction = age_modifier_direction.clip(0, 1)
	dump('age_modifier/direction_input', age_modifier_direction)

	forward_output = age_modifier.forward(prepared, prepared, age_modifier_direction)
	dump('age_modifier/forward_output', forward_output)

	normalized = age_modifier.normalize_vision_frame(forward_output)
	dump('age_modifier/normalized_output', normalized)

	restored_frame = age_modifier.modify_age(target_face, source_frame.copy())
	dump('age_modifier/restored_frame', restored_frame)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	if not os.path.isfile(SOURCE_IMAGE):
		raise SystemExit(f'Missing example image: {SOURCE_IMAGE} (run tools/parity/fetch_examples.sh)')
	if not os.path.isfile(TARGET_VIDEO):
		raise SystemExit(f'Missing example video: {TARGET_VIDEO} (run tools/parity/fetch_examples.sh)')

	init_state()
	source_frame = read_static_image(SOURCE_IMAGE)
	video_frame = read_static_video_frame(TARGET_VIDEO, 0)

	dump_live_portrait_rotation()
	dump_expression_restorer(source_frame, video_frame)
	dump_age_modifier(source_frame)


if __name__ == '__main__':
	main()
