"""
Dump ground truth from `facefusion.processors.modules.face_swapper.core` for the C#
`FaceFusion.Processors.FaceSwapper` parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/face_swapper` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/face_swapper \
        python3 tools/parity/dump_face_swapper.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg, target-240p.mp4) and a working internet
connection the first time it runs, since `pre_check()` downloads the swapper/recognizer/
detector models into `.assets/models/` if not already present (inswapper_128 is ~530MB,
ghost_1_256 + its crossface_ghost embedding converter together are ~510MB).

Kept lean per the assignment: two model families exercise the two distinct source-input
paths `forward_swap_face` takes (`inswapper` -> `get_static_model_initializer` dot product,
`ghost` -> a real `embedding_converter` ONNX pass + the renormalize branch of
`normalize_crop_frame`), one source/target frame pair, `face_swapper_pixel_boost` pinned to
each model's own native size (`pixel_boost_total == 1`) so pixel-boost tiling itself is
verified separately in pure C# (`PixelBoostTests`, no fixture needed — it is a closed-form
reshape/transpose), and `face_mask_types = ['box']` only (no occlusion/area/region ONNX
session needed; those mask functions are `FaceFusion.Face.FaceMasker`'s own, already
parity-tested there — reused, not re-verified, by face_swapper).

For each family the highest-value dumps are the exact model input tensors
(`.../target_input.npy`, `.../source_input.npy`) handed to `session.run` — if those match
the C# port's preprocessing output, any further mismatch is ONNX Runtime's own arithmetic.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
from facefusion.face_creator import average_face_identity, get_static_faces, scale_face
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask
from facefusion.face_selector import select_faces, sort_faces_by_order
from facefusion.model_helper import get_static_model_initializer
from facefusion.processors.modules.face_swapper import core as face_swapper
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.vision import read_static_image, unpack_resolution
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')
TARGET_IMAGE = os.path.join(EXAMPLES_DIR, 'target-240p.mp4')


def init_state(face_swapper_model: str, pixel_boost: str) -> None:
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
		('face_selector_mode', 'reference'),
		('face_selector_order', 'large-small'),
		('face_selector_gender', None),
		('face_selector_race', None),
		('face_selector_age_start', None),
		('face_selector_age_end', None),
		('reference_face_position', 0),
		('reference_face_distance', 0.6),
		('face_tracker_score', 0.0),
		('face_mask_types', ['box']),
		('face_mask_blur', 0.3),
		('face_mask_padding', (0, 0, 0, 0)),
		('face_swapper_model', face_swapper_model),
		('face_swapper_pixel_boost', pixel_boost),
		('face_swapper_weight', 0.5),
	]:
		state_manager.init_item(key, value)


def dump_family(family_name: str, face_swapper_model: str) -> None:
	init_state(face_swapper_model, '256x256')
	model_size = face_swapper.create_static_model_set('full').get(face_swapper_model).get('size')
	pixel_boost = f'{model_size[0]}x{model_size[1]}'
	state_manager.set_item('face_swapper_pixel_boost', pixel_boost)

	if not face_swapper.pre_check():
		raise SystemExit(f'face_swapper.pre_check() failed to fetch models for {face_swapper_model}.')

	source_frame = read_static_image(SOURCE_IMAGE)
	source_faces = get_static_faces([source_frame])
	source_faces = sort_faces_by_order(source_faces, 'large-small')
	source_face = average_face_identity([source_faces[0]])
	dump(f'{family_name}/source_bounding_box', numpy.asarray(source_face.bounding_box))
	dump(f'{family_name}/source_embedding', source_face.embedding)
	dump(f'{family_name}/source_embedding_norm', source_face.embedding_norm)

	# reference frame == source frame == target frame keeps this a single-image, one-face
	# fixture (no video decode needed to reproduce it) while still exercising the exact same
	# select_faces()/scale_face()/swap_face() call sequence process_frame() uses.
	target_faces = select_faces(source_frame, [source_frame], [source_frame])
	target_face = target_faces[0]
	target_face = scale_face(target_face, source_frame, source_frame)
	dump(f'{family_name}/target_bounding_box', numpy.asarray(target_face.bounding_box))
	dump(f'{family_name}/target_landmark_5_of_68', target_face.landmark_set.get('5/68'))
	dump(f'{family_name}/target_embedding', target_face.embedding)

	model_options = face_swapper.get_model_options()
	model_template = model_options.get('template')
	model_mean = model_options.get('mean')
	model_standard_deviation = model_options.get('standard_deviation')

	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(source_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	dump(f'{family_name}/crop_vision_frame', crop_vision_frame)

	box_mask = create_box_mask(crop_vision_frame, 0.3, (0, 0, 0, 0))
	dump(f'{family_name}/box_mask', box_mask)

	# --- target_input: prepare_crop_frame's output, the 'target' model input tensor ---
	target_input = crop_vision_frame[:, :, ::-1] / 255.0
	target_input = (target_input - model_mean) / model_standard_deviation
	target_input = target_input.transpose(2, 0, 1)
	target_input = numpy.expand_dims(target_input, axis=0).astype(numpy.float32)
	dump(f'{family_name}/target_input', target_input)

	# --- source_input: the 'source' model input (family-specific path) ---
	model_type = model_options.get('type')
	if model_type == 'inswapper':
		model_path = model_options.get('sources').get('face_swapper').get('path')
		model_initializer = get_static_model_initializer(model_path)
		source_embedding = source_face.embedding.reshape((1, -1))
		source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
		dump(f'{family_name}/model_initializer', model_initializer)
	elif model_type == 'ghost':
		source_embedding = source_face.embedding.reshape(-1, 512)
		source_embedding, _ = face_swapper.convert_source_embedding(source_embedding)
		source_embedding = source_embedding.reshape(1, -1)
	else:
		raise SystemExit(f'dump_face_swapper.py only supports inswapper/ghost, got {model_type}')

	dump(f'{family_name}/prepared_source_embedding', source_embedding)

	source_embedding = face_swapper.balance_source_embedding(source_embedding, target_face.embedding)
	dump(f'{family_name}/source_input', source_embedding.astype(numpy.float32))

	# Not dumped: a full swap_face()/paste_back() end-to-end frame. paste_back's geometry is
	# FaceFusion.Face.FaceHelper's own (already parity-tested there via
	# FaceAnalysisParityTests' warp tests) and the blend arithmetic is generic managed float
	# math with no model-specific behaviour; re-dumping a multi-megabyte full source.jpg-sized
	# frame here would not exercise anything box_mask/crop_vision_frame/normalized_crop_frame
	# below do not already cover, at several MB of fixture cost (see PARITY_HARNESS.md's
	# "keep fixtures lean" / gzip-over-500KB guidance).

	# raw model output (pre normalize_crop_frame), captured via a second forward pass so the
	# normalize step itself is isolated and independently checkable
	inference_pool = face_swapper.get_inference_pool()
	face_swapper_session = inference_pool.get('face_swapper')
	session_inputs = {}
	for model_input in face_swapper_session.get_inputs():
		if model_input.name == 'source':
			session_inputs[model_input.name] = source_embedding.astype(numpy.float32)
		if model_input.name == 'target':
			session_inputs[model_input.name] = target_input
	raw_output = face_swapper_session.run(None, session_inputs)[0][0]
	dump(f'{family_name}/raw_model_output', raw_output)

	normalized = raw_output.transpose(1, 2, 0)
	if model_type in ('ghost', 'hififace', 'hyperswap', 'uniface'):
		normalized = normalized * model_standard_deviation + model_mean
	normalized = normalized.clip(0, 1)
	normalized = normalized[:, :, ::-1] * 255
	dump(f'{family_name}/normalized_crop_frame', normalized)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	dump_family('inswapper_128', 'inswapper_128')
	dump_family('ghost_1_256', 'ghost_1_256')


if __name__ == '__main__':
	main()
