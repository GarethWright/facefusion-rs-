"""
Dump ground truth from the real `facefusion.face_detector` for the C# FaceDetector parity
tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/face_detector` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/face_detector \
        python3 tools/parity/dump_face_detector.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg, target-240p.mp4) and a working internet
connection the first time it runs, since `pre_check()` downloads the four detector ONNX
models (~35 MB total) into `.assets/models/` if they are not already present.

Kept lean per the assignment: one image (source.jpg) drives the per-family cases, plus one
video frame case. The highest-value dumps are `model_input/source_640x640/*.npy` — the exact
tensors handed to `session.run`, dumped once per distinct (frame, size, normalize_range)
combination rather than once per family, since `prepare_detect_frame` does not depend on the
family and three of the four families share a normalize_range with another family or with the
identity case:
  - raw.npy            = prepare_detect_frame() output, unnormalized -> yunet's model input
                          (normalize_range [0, 255] is a no-op, see normalize_detect_frame)
  - normalized_m1_1.npy = normalize_detect_frame(..., [-1, 1])        -> retinaface + scrfd
  - normalized_0_1.npy  = normalize_detect_frame(..., [0, 1])         -> yolo_face

If these tensors match the C# port's `FaceDetector.PrepareDetectFrame` +
`FaceDetector.NormalizeDetectFrame` output, any further mismatch against the bounding
box/score/landmark fixtures is ONNX Runtime's own arithmetic (expect ~0 divergence there per
PARITY_HARNESS.md); if the tensors do not match, the bug is in the .NET preprocessing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from facefusion import face_detector, state_manager
from facefusion.vision import read_static_image, read_static_video_frame, restrict_frame
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')
TARGET_240P = os.path.join(EXAMPLES_DIR, 'target-240p.mp4')

FAMILIES = ('retinaface', 'scrfd', 'yolo_face', 'yunet')

DETECT_FN_BY_FAMILY = {
	'retinaface': face_detector.detect_with_retinaface,
	'scrfd': face_detector.detect_with_scrfd,
	'yolo_face': face_detector.detect_with_yolo_face,
	'yunet': face_detector.detect_with_yunet,
}


def init_state() -> None:
	for key, value in [
		('download_providers', ['github']),
		('execution_providers', ['cpu']),
		('execution_device_ids', [0]),
		('face_detector_score', 0.5),
		('face_detector_angles', [0]),
		('face_detector_margin', [0, 0, 0, 0]),
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
	]:
		state_manager.init_item(key, value)


def dump_model_input_tensors(name: str, vision_frame, face_detector_size: str) -> None:
	"""The de-duplicated model-input tensor set — see module docstring."""
	face_detector_width, face_detector_height = face_detector.unpack_resolution(face_detector_size)
	temp_vision_frame = restrict_frame(vision_frame, (face_detector_width, face_detector_height))
	raw = face_detector.prepare_detect_frame(temp_vision_frame, face_detector_size)

	dump(f'{name}/raw', raw)
	dump(f'{name}/normalized_m1_1', face_detector.normalize_detect_frame(raw, [-1, 1]))
	dump(f'{name}/normalized_0_1', face_detector.normalize_detect_frame(raw, [0, 1]))


def dump_family_outputs(family: str, vision_frame, prefix: str, face_detector_size: str) -> None:
	state_manager.init_item('face_detector_model', family)
	state_manager.init_item('face_detector_size', face_detector_size)

	bounding_boxes, face_scores, face_landmarks_5 = DETECT_FN_BY_FAMILY[family](vision_frame, face_detector_size)
	dump(f'{prefix}/bounding_boxes', bounding_boxes)
	dump(f'{prefix}/face_scores', face_scores)
	dump(f'{prefix}/face_landmarks_5', face_landmarks_5)

	# End-to-end detect_faces() (adds the normalize_bounding_box + margin-shift step on top
	# of detect_with_<family>) — exercises prepare_margin() too, with the default zero margin.
	all_bounding_boxes, all_face_scores, all_face_landmarks_5 = face_detector.detect_faces(vision_frame)
	dump(f'{prefix}/detect_faces_bounding_boxes', all_bounding_boxes)
	dump(f'{prefix}/detect_faces_face_scores', all_face_scores)
	dump(f'{prefix}/detect_faces_face_landmarks_5', all_face_landmarks_5)


def dump_prepare_margin_cases(vision_frame) -> None:
	cases = [
		[0, 0, 0, 0],
		[10, 5, 20, 15],
	]
	results = []

	for margin in cases:
		state_manager.init_item('face_detector_margin', margin)
		margin_top, margin_right, margin_bottom, margin_left = face_detector.prepare_margin(vision_frame)
		results.append({
			'margin': margin,
			'result': [margin_top, margin_right, margin_bottom, margin_left],
		})

	state_manager.init_item('face_detector_margin', [0, 0, 0, 0])
	dump_scalar('prepare_margin_cases', results)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	init_state()

	# 'many' is enough to trigger downloads for all four families in one pre_check() call.
	state_manager.init_item('face_detector_model', 'many')
	if not face_detector.pre_check():
		raise SystemExit('face_detector.pre_check() failed to fetch the detector models.')

	source_frame = read_static_image(SOURCE_IMAGE)
	dump_prepare_margin_cases(source_frame)

	# The one shared model-input tensor set (source.jpg, 640x640) that every family's
	# detect_frame_input case is checked against in C#.
	dump_model_input_tensors('model_input/source_640x640', source_frame, '640x640')

	for family in FAMILIES:
		dump_family_outputs(family, source_frame, f'{family}/source_640x640', '640x640')

	# A second face_detector_size exercises the restrict_frame ratio-scaling path in
	# detect_with_retinaface (ratio_width/ratio_height != 1) — outputs only, no tensor dump
	# (the preprocessing itself is already covered by the 640x640 case above).
	dump_family_outputs('retinaface', source_frame, 'retinaface/source_320x320', '320x320')

	# One video frame case, end to end. The frame pixels are dumped too so the C# test can
	# load the exact Python-decoded bytes rather than re-decoding the video itself (see
	# VisionParityTests' documented OpenCvSharp.VideoCapture-vs-ffmpeg decode divergence) —
	# isolating this case to genuine detector-math parity, not decode-path noise.
	video_frame = read_static_video_frame(TARGET_240P, 0)
	dump('video/target_240p_frame_0', video_frame)
	dump_family_outputs('yolo_face', video_frame, 'yolo_face/video_frame_0_640x640', '640x640')


if __name__ == '__main__':
	main()
