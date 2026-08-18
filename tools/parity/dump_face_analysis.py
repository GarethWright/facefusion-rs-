"""
Dump ground truth from `facefusion.face_landmarker`, `facefusion.face_recognizer` and
`facefusion.face_classifier` for the C# FaceLandmarker/FaceRecognizer/FaceClassifier parity
tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/face_analysis` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/face_analysis \
        python3 tools/parity/dump_face_analysis.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg) and a working internet connection the first
time it runs, since `pre_check()` downloads the landmarker/recognizer/classifier ONNX models
into `.assets/models/` if they are not already present.

Kept lean per the assignment: one face (the largest detected face on source.jpg) drives every
case. For each of the three modules the highest-value dump is the exact model input tensor
(`.../input.npy`) handed to `session.run` — if that tensor matches the C# port's
preprocessing output, any further mismatch is ONNX Runtime's own arithmetic (expect ~0
divergence there per PARITY_HARNESS.md); if it does not match, the bug is in the .NET
preprocessing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import face_classifier, face_detector, face_landmarker, face_recognizer, state_manager
from facefusion.face_helper import convert_to_face_landmark_5
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
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
		('face_landmarker_model', 'many'),
		('face_landmarker_score', 0.0),
	]:
		state_manager.init_item(key, value)


def pick_reference_face(vision_frame):
	"""The largest detected face on source.jpg, by bounding-box area — deterministic and not
	dependent on detector ordering."""
	bounding_boxes, face_scores, face_landmarks_5 = face_detector.detect_faces(vision_frame)
	areas = [ (box[2] - box[0]) * (box[3] - box[1]) for box in bounding_boxes ]
	index = int(numpy.argmax(areas))
	return bounding_boxes[index], face_scores[index], face_landmarks_5[index]


def dump_face_landmarker(vision_frame, bounding_box, face_landmark_5) -> None:
	# estimate_face_angle expects a 68-point landmark set (indices 0 and 16); it is only ever
	# called downstream (face_creator, out of this assignment's scope) on a real 68-point
	# landmark set derived from a *previous* landmarker pass. For this dump the angle input to
	# detect_with_2dfan4/detect_with_peppa_wutz is exercised directly at angle = 0, the value
	# every real pipeline run uses today (face_detector_angles defaults to [0]).
	face_angle = 0

	dump('face_landmarker/bounding_box', bounding_box)
	dump('face_landmarker/face_angle', numpy.asarray(face_angle))

	model_size = face_landmarker.create_static_model_set('full').get('2dfan4').get('size')
	scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
	translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
	dump_scalar('face_landmarker/2dfan4_scale', float(scale))
	dump('face_landmarker/2dfan4_translation', translation)

	from facefusion.face_helper import create_rotation_matrix_and_size, warp_face_by_translation
	import cv2

	rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
	crop_vision_frame, affine_matrix = warp_face_by_translation(vision_frame, translation, scale, model_size)
	crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
	# Pre-contrast crop, dumped so the contrast step (conditional_optimize_contrast) can be
	# parity-tested in isolation from the translate+rotate warp that produced it.
	dump('face_landmarker/pre_contrast_crop', crop_vision_frame)
	crop_vision_frame = face_landmarker.conditional_optimize_contrast(crop_vision_frame)
	dump('face_landmarker/optimized_contrast_crop', crop_vision_frame)
	model_input = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
	# The real forward_with_2dfan4 wraps the crop in a python list (`{'input': [crop]}`),
	# which onnxruntime turns into a (1, 3, 256, 256) batch tensor identical to an explicit
	# expand_dims — dumped that way so the fixture is directly the model's input shape.
	# forward_with_peppa_wutz's own expand_dims produces the exact same (1, 3, 256, 256)
	# tensor from the exact same crop (both models share model_size and preprocessing), so
	# one fixture covers the model-input tensor for both families.
	dump('face_landmarker/model_input', numpy.expand_dims(model_input, axis = 0))

	face_landmark_68_2dfan4, face_landmark_score_2dfan4 = face_landmarker.detect_with_2dfan4(vision_frame, bounding_box, face_angle)
	dump('face_landmarker/2dfan4_landmark_68', face_landmark_68_2dfan4)
	dump_scalar('face_landmarker/2dfan4_score', float(face_landmark_score_2dfan4))

	face_landmark_68_peppa, face_landmark_score_peppa = face_landmarker.detect_with_peppa_wutz(vision_frame, bounding_box, face_angle)
	dump('face_landmarker/peppa_wutz_landmark_68', face_landmark_68_peppa)
	dump_scalar('face_landmarker/peppa_wutz_score', float(face_landmark_score_peppa))

	face_landmark_68_final, face_landmark_score_final = face_landmarker.detect_face_landmark(vision_frame, bounding_box, face_angle)
	dump('face_landmarker/detect_face_landmark_68', face_landmark_68_final)
	dump_scalar('face_landmarker/detect_face_landmark_score', float(face_landmark_score_final))

	face_landmark_5_68 = convert_to_face_landmark_5(face_landmark_68_final)
	face_landmark_68_5 = face_landmarker.estimate_face_landmark_68_5(face_landmark_5_68)
	dump('face_landmarker/fan_68_5_input', numpy.asarray([ face_landmark_5_68 ], dtype = numpy.float32))
	dump('face_landmarker/fan_68_5_output', face_landmark_68_5)


def dump_face_recognizer(vision_frame, face_landmark_5) -> None:
	from facefusion.face_helper import warp_face_by_face_landmark_5

	model_template = face_recognizer.get_model_options().get('template')
	model_size = face_recognizer.get_model_options().get('size')
	crop_vision_frame, _ = warp_face_by_face_landmark_5(vision_frame, face_landmark_5, model_template, model_size)
	dump('face_recognizer/crop_vision_frame', crop_vision_frame)

	model_input = crop_vision_frame / 127.5 - 1
	model_input = model_input[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
	model_input = numpy.expand_dims(model_input, axis = 0)
	dump('face_recognizer/input', model_input)

	face_embedding, face_embedding_norm = face_recognizer.calculate_face_embedding(vision_frame, face_landmark_5)
	dump('face_recognizer/embedding', face_embedding)
	dump('face_recognizer/embedding_norm', face_embedding_norm)


def dump_face_classifier(vision_frame, face_landmark_5) -> None:
	from facefusion.face_helper import warp_face_by_face_landmark_5

	model_template = face_classifier.get_model_options().get('template')
	model_size = face_classifier.get_model_options().get('size')
	model_mean = face_classifier.get_model_options().get('mean')
	model_standard_deviation = face_classifier.get_model_options().get('standard_deviation')
	crop_vision_frame, _ = warp_face_by_face_landmark_5(vision_frame, face_landmark_5, model_template, model_size)
	dump('face_classifier/crop_vision_frame', crop_vision_frame)

	model_input = crop_vision_frame.astype(numpy.float32)[:, :, ::-1] / 255.0
	model_input -= model_mean
	model_input /= model_standard_deviation
	model_input = model_input.transpose(2, 0, 1)
	model_input = numpy.expand_dims(model_input, axis = 0)
	dump('face_classifier/input', model_input)

	gender_id, age_id, race_id = face_classifier.forward(model_input)
	dump('face_classifier/race_id_raw', race_id)
	dump('face_classifier/gender_id_raw', gender_id)
	dump('face_classifier/age_id_raw', age_id)

	gender, age, race = face_classifier.classify_face(vision_frame, face_landmark_5)
	dump_scalar('face_classifier/gender', gender)
	dump_scalar('face_classifier/age', [ age.start, age.stop ])
	dump_scalar('face_classifier/race', race)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	init_state()

	if not face_detector.pre_check():
		raise SystemExit('face_detector.pre_check() failed to fetch the detector models.')
	if not face_landmarker.pre_check():
		raise SystemExit('face_landmarker.pre_check() failed to fetch the landmarker models.')
	if not face_recognizer.pre_check():
		raise SystemExit('face_recognizer.pre_check() failed to fetch the recognizer model.')
	if not face_classifier.pre_check():
		raise SystemExit('face_classifier.pre_check() failed to fetch the classifier model.')

	source_frame = read_static_image(SOURCE_IMAGE)
	bounding_box, face_score, face_landmark_5 = pick_reference_face(source_frame)
	dump('reference/bounding_box', bounding_box)
	dump('reference/face_score', numpy.asarray(face_score))
	dump('reference/face_landmark_5', face_landmark_5)

	dump_face_landmarker(source_frame, bounding_box, face_landmark_5)
	dump_face_recognizer(source_frame, face_landmark_5)
	dump_face_classifier(source_frame, face_landmark_5)


if __name__ == '__main__':
	main()
