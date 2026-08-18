"""
Dump ground truth from `facefusion.processors.modules.face_enhancer` for the C#
FaceEnhancer parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/face_enhancer` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/face_enhancer \
        python3 tools/parity/dump_face_enhancer.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg) and a working internet connection the first
time it runs, since `pre_check()` downloads the yolo_face detector and gpen_bfr_256 face
enhancer ONNX models into `.assets/models/` if not already present.

Kept lean per the assignment: one face (the largest detected face on source.jpg) and one
model family (gpen_bfr_256 — the smallest face_enhancer model, no 'weight' input) drive
every case. The highest-value dump is the exact model input tensor
(`model_input/gpen_bfr_256/input.npy`) handed to `session.run` — if that tensor matches the
.NET port's preprocessing output, any further mismatch is ONNX Runtime's own arithmetic
(expect ~0 divergence per PARITY_HARNESS.md); if it does not match, the bug is in .NET
preprocessing (FaceEnhancer.PrepareCropFrame).

Note: every 'weight'-input model (codeformer) is 300+ MB, too large for this lean fixture
set; FaceEnhancer.Forward's weight-input branch is implemented (loops
`InputNames` the same way Python loops `get_inputs()`) but is not exercised by a committed
fixture — see the port report.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import face_detector, state_manager
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask
from facefusion.processors.modules.face_enhancer import core as face_enhancer
from facefusion.vision import read_static_image
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')

MODEL_NAME = 'gpen_bfr_256'


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
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
		('face_enhancer_model', MODEL_NAME),
		('face_enhancer_blend', 80),
		('face_enhancer_weight', 0.5),
		('face_mask_blur', 0.3),
		('face_mask_types', ['box']),
		('face_landmarker_model', 'many'),
		('face_landmarker_score', 0.0),
	]:
		state_manager.init_item(key, value)


def pick_reference_face(vision_frame):
	"""The largest detected face on source.jpg, by bounding-box area — deterministic and not
	dependent on detector ordering."""
	bounding_boxes, face_scores, face_landmarks_5 = face_detector.detect_faces(vision_frame)
	areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bounding_boxes]
	index = int(numpy.argmax(areas))
	return bounding_boxes[index], face_scores[index], face_landmarks_5[index]


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	init_state()

	if not face_detector.pre_check():
		raise SystemExit('face_detector.pre_check() failed to fetch the detector models.')
	if not face_enhancer.pre_check():
		raise SystemExit('face_enhancer.pre_check() failed to fetch the face_enhancer model.')

	source_frame = read_static_image(SOURCE_IMAGE)
	bounding_box, face_score, face_landmark_5 = pick_reference_face(source_frame)
	dump('reference/bounding_box', bounding_box)
	dump('reference/face_landmark_5', face_landmark_5)

	model_template = face_enhancer.get_model_options().get('template')
	model_size = face_enhancer.get_model_options().get('size')
	dump_scalar('reference/model_size', list(model_size))

	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(source_frame, face_landmark_5, model_template, model_size)
	dump(MODEL_NAME + '/crop_vision_frame', crop_vision_frame)
	dump(MODEL_NAME + '/affine_matrix', affine_matrix)

	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
	dump(MODEL_NAME + '/box_mask', box_mask)

	prepared_crop_vision_frame = face_enhancer.prepare_crop_frame(crop_vision_frame)
	dump(MODEL_NAME + '/input', prepared_crop_vision_frame)

	face_enhancer_weight = numpy.array([state_manager.get_item('face_enhancer_weight')]).astype(numpy.double)
	forward_output = face_enhancer.forward(prepared_crop_vision_frame, face_enhancer_weight)
	dump(MODEL_NAME + '/forward_output', forward_output)

	normalized_crop_vision_frame = face_enhancer.normalize_crop_frame(forward_output)
	dump(MODEL_NAME + '/normalized_crop_vision_frame', normalized_crop_vision_frame)

	# paste_back()/blend_paste_frame() are exercised end to end below via enhance_face()
	# rather than dumped as separate intermediate frames — both are thin, already
	# individually-tested (FaceHelperTests/VisionParityTests) wrappers, and the full-frame
	# .npy they'd produce here is identical in size to enhance_face_output below, so a
	# separate dump would only add redundant fixture weight without adding coverage.

	# End-to-end enhance_face(), for an image-level (PSNR/SSIM) sanity check in addition to
	# the stage-by-stage tensors above.
	from facefusion.face_creator import create_faces

	faces = create_faces(source_frame, *face_detector.detect_faces(source_frame))
	target_face = max(faces, key=lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]))
	enhanced_frame = face_enhancer.enhance_face(target_face, source_frame)
	dump(MODEL_NAME + '/enhance_face_output', enhanced_frame)


if __name__ == '__main__':
	main()
